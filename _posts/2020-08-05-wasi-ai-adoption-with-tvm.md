---
layout: post
title:  "WASM AI Adoption with TVM Stack"
date:   2020-08-05 11:16:56 +0800
categories: machine-learning
---

We introduced a new solution that compiles machine learning models into WASM bytecode using TVM, an end to end deep learning compiler stack, and loads the WASM graph in runtime environment using Wasmtime.

- [Background](#background)
    - [Investigation](#investigation)
    - [Preliminary study](#preliminary-study)
- [Solution Introduction](#solution-introduction)
    - [Design](#design)
    - [Code implementation](#code-implementation)
    - [PoC guideline](#poc-guideline)
- [Solution Promotion](#solution-promotion)
    - [2020.06 - Incubation](#2020.06---incubation)
    - [2020.07 - TVM community promotion](#2020.07---tvm-community-promotion)
- [Future Plan](#future-plan)
    - [TVM community collaboration](#tvm-community-collaboration)
    - [WASM operator library study](#wasm-operator-library-study)
- [Acknowledgement](#acknowledgement)

## Background

### Investigation

It's all known that [WebAssembly](https://webassembly.org/) has been playing an important role in web scenario, and it has already been shipped in 4 major browser engines. But with its powerful features of portability, it has gained more and more attention in non-web scenarios, especially in IoT and mobile devices.

As a research team in AI field, we are so excited that there already are some great ideas about landing WASM in AI field:
* TensorFlow community released a [blogpost](https://blog.tensorflow.org/2020/03/introducing-webassembly-backend-for-tensorflow-js.html) regarding the support of WebAssembly as Tensorflow.js backend.
* TVM community released a [blogpost](https://tvm.apache.org/2020/05/14/compiling-machine-learning-to-webassembly-and-webgpu) to introducing support for WASM and WebGPU to the Apache TVM deep learning compiler.
* [XNNPACK](https://github.com/google/XNNPACK) leveraged the portability feature of WASM to execute operators on any hardware devices.

### Preliminary study

As the [blogpost](https://medium.com/@nopainkiller/what-wasi-wasm-could-bring-new-exciting-features-to-ai-e04e303849b2) that we released shown, we also expect to leverage the portability of WASM so as to solve the problem of AI model landing in all scenarios. Currently for models defined for traditional deep learning frameworks, users have to perform additional customized development work when performing model training/inference on different hardware environments, sometimes even need to develop a set of inference engine systems separately.

So how to leverage the portability of WASM to realize the unification of the hardware environment? Take the [MindSpore](https://www.mindspore.cn/) deep learning framework as an example, if we analyze the MindSpore model from the macro and micro perspectives separately, the macro view is a **computing graph** based on the MindSpore IR definition, and the micro view it is the collection with a series of MindSpore **operators**. Then we can try to combine WASM with the deep learning framework from the dimensions of computing graph and operator respectively, that is, to propose the two concepts of `WASM computing graph` and `WASM operator library`.

* WASM computing graph

    WASM computing graph, as the name implies, is to compile the trained model (including model parameters) into WASM bytecode, and then load it in the Runtime environment through WASM Runtime for performing model inference. With the help of WASM portability, it can be implemented in any environment Model inference work:
    * In the web scenario, the WASM bytecode can be loaded into the JS Runtime through the `Emscripten` tool and then executed in the browser.
    * In non-web scenarios, use the `Wasmtime` tool to load WASM bytecode into the system environment for execution.

    In the case of the WASM calculation graph, since the trained models (and parameters) are saved in the system environment in advance, it is necessary to introduce the `WASI` interface to interact with system resources to complete the operation of offline loading models. Therefore, when choosing WASM Runtime, you need to choose a tool that supports the WASI (WASM System Interface) standard (such as Wasmtime), or you can directly extend Emscripten with WASI like what the TVM community did.

* WASM operator library

    The WASM operator library is relatively easy to understand, which is to compile a single operator into WASM bytecode, and then provide an encapsulated operator call interface for the upper framework. But unlike the traditional handwriting operator, the framework needs to load the WASM operator in a way similar to dynamic linking. However, considering that the current WASM itself only support static linking, it is necessary to pre-load all compiled WASM The operators are integrated, and then the call interface of the operator library is provided to the framework layer.

## Solution Introduction

Based on the analysis and comparison of two ideas shown above, and at the same time, with reference to the existing work from TVM community, we decided to start with the in-depth exploration of the road of `WASM computing graph`, and reuse the full stack compilation of TVM to quickly realize the prototype of the solution.

**NOTICE:** If you are interested into TVM project, please refer to [TVM Community](https://tvm.apache.org/) for the details.

### Design

Since the overall design idea has been illustrated in the background section, this chapter is mainly to indicate how to use the TVM compilation stack to achieve the whole compilation process. The figures below demonstrate the whole landscape of running deep learning frameworks on WASM runtime with TVM compiler stack.

* WASM graph generation
    ```
       _ _ _ _ _ _ _ _ _ _        _ _ _ _ _ _ _        _ _ _ _ _ _ _ _ _ _ _ _
      |                   |      |             |      |                       |
      |  MindSpore Model  | ---> |  ONNX Model | ---> |  TVM Relay Python API |
      |_ _ _ _ _ _ _ _ _ _|      |_ _ _ _ _ _ _|      |_ _ _ _ _ _ _ _ _ _ _ _|
                                                                 ||
                                                                 \/
                 _ _ _ _ _ _ _ _ _ _ _                  _ _ _ _ _ _ _ _ _ _ _
                |                     |                |                     |
                | WASM Graph Builder  |                |  TVM Compiler Stack |
                |    (TVM runtime)    |                |_ _ _ _ _ _ _ _ _ _ _|
                |_ _ _ _ _ _ _ _ _ _ _|                          ||
                          ||                                     \/
      _ _ _ _ _ _ _ _ _   ||   _ _ _ _ _ _ _ _ _ _            _ _ _ _ _
     |                 |  \/  |                   |  llvm-ar |         |
     | wasm_graph.wasm | <--- | libgraph_wasm32.a | <------- | graph.o |
     |_ _ _ _ _ _ _ _ _|      |_ _ _ _ _ _ _ _ _ _|          |_ _ _ _ _|
    ```
    As shown in the figure above, we can use the Python interface of TVM Relay to directly compile the model into an executable file of `graph.o`, but it should be noted that the generated graph.o file cannot be directly recognized by the WASM runtime module. It's required to be loaded through TVM Rust runtime and then use Rust compiler (with `wasm32-wasi` target) to directly compile the `WASM Graph Builder` module into WASM bytecode (that is, the `wasm_graph.wasm` file in the figure). Why do we have to go through this tedious step of conversion? Mainly because the `graph.o` file contains the primitives of Relay and TVM IR, we cannot directly convert these primitives into WASM primitives.

* WASM graph loading
    ```
         _ _ _ _ _ _ _ _ _ _ _
        |                     |
        |  WASM Graph Loader  |
        |   (WASM runtime)    |
        |_ _ _ _ _ _ _ _ _ _ _|
                  ||
                  \/
          _ _ _ _ _ _ _ _ _ _
         |                   |
         |  wasm_graph.wasm  |
         |_ _ _ _ _ _ _ _ _ _|
    ```
    The graph loading phase (from the figure above) seems to be very simple, but actually it's far more complicated. First of all, WASM's runtime defines a set of assembly-level user interfaces for WASM IR, which is extremely unfriendly to end-user developers; secondly. Secondly, WASM currently only supports integer types (such as `i32`, `u64`, etc.) as function parameters, which makes the `Tensor` types defined in the deep learning field unable to be passed in natively. Besides, it's not to be mentioned to support WASM advanced features such as `thread` and `SIMD128`.

    Of course, the exploration of each new field is inseparable from a variety of problems, and solving the problems itself is the job of the technical/scientific personnel, so we did not rely on the WASM community but actively tried to solve these problems. Since current WASM interface is not designed for the high-level API for upper users, we can self-develop a set of interfaces based on our own needs. Although WASM does not support passing in `struct` or `pointer`, we could firstly serialize Tensor data and write it to WASM linear memory through the WASM Memory mechanism, and then convert the memory address to i32 type as function parameters. Although some changes are a bit "anti-human", they can clearly show our thoughts and ideas, which should be enough.

### Code implementation

Here we introduce the codebase of this project:
```
wasm-standalone/
├── README.md
├── wasm-graph      // WASM graph generation module
│   ├── build.rs    // build script
│   ├── Cargo.toml  // project dependencies
│   ├── lib         // store the computing graph compiled by TVM Relay API
│   │   ├── graph.json
│   │   ├── graph.o
│   │   ├── graph.params
│   │   └── libgraph_wasm32.a
│   ├── src         // WASM graph generation source code
│   │   ├── lib.rs
│   │   ├── types.rs
│   │   └── utils.rs
│   └── tools       // Relay Python API compilation script
│       ├── build_graph_lib.py
└── wasm-runtime    // WASM graph loading module
    ├── Cargo.toml
    ├── src         // WASM graph loading source code
    │   ├── graph.rs
    │   ├── lib.rs
    │   └── types.rs
    └── tests      // WASM graph loading test module
        └── test_graph_resnet50
```

For more details, please take a look at the code repo (https://github.com/leonwanghui/ms-backend-wasm/tree/master/wasm-standalone-tvm).

**2020.07 Update:** Considering this solution has been accepted by TVM community, it's suggested for users to try with the official one (https://github.com/apache/incubator-tvm/tree/master/apps/wasm-standalone).

### PoC guideline

In order for everyone to have a more vivid and concrete understanding of the solution, we prepared a simple prototype: compile the ResNet50 model generated based on ONNX into a `wasm_graph_resnet50.wasm` file through the TVM Relay API, and then call Wasmtime tool to load WASM bytecode in the runtime environment so as to perform model inference. For the detailed guidelines, please refer to the [online document](https://github.com/apache/incubator-tvm/blob/master/apps/wasm-standalone/README.md#poc-guidelines).

## Solution Promotion

As a development team that has been working in the open source community over four years, we have developed four Open habits (`Open Source`, `Open Design`, `Open Development` and `Open Community`). This chapter will take you to review our promotion about the journey of WASM adoption in AI.

### 2020.06 - Incubation

For the motto that has been popular in the open source community: "Talk is cheap, show me the code", it is necessary to prepare the PoC for the promotion stage. Therefore, it took us less than a month to finish the work with development of the prototype, which laid the good basis for our later promotion in the TVM community.

### 2020.07 - TVM community promotion

Since the solution is based on the TVM compilation stack, our first idea after incubation stage is to give it back to the TVM community, and then self-recommend as the owner for later feature development and maintenance. After continuous promotion in the community over the past month, with the initial proposal gone through a series of reconstruction and rectification, and finally adopted by the community at the end of July. If you are interested into more about this proposal, please refer to community PR[#5892](https://github.com/apache/incubator-tvm/pull/5892) for details.

## Future Plan

### TVM community collaboration

As mentioned above, the project is still in the **experimental** stage, so we will continuously work with the TVM community to explore more possibilities. The current preliminary planning features are listed below:
* Support data parallel processing based on SIMD128.
* Further improve the TVM Rust runtime API module so that it can natively support the docking of WASM Memory features.
* AutoTVM optimization based on WASM backend.
* More network support.

### WASM operator library study

At present, we have only conducted in-depth exploration in the direction of WASM computing graphs, but we found that it would more powerful to integrate WASM operator library into deep learning frameworks (such as MindSpore). Here are a few more suitable scenarios for the WASM operator library:
* Currently deep learning frameworks have defined their own IR and compilation pipelines. Compared with WASM computing graph, only WASM operator library can seamlessly integrate with the graph computing layer of these frameworks.
* The WASM computing graph can only be used for model inference, while the WASM operator library can be applied to all scenarios (such as model training/inference/evaluation).
* As for the portability feature, the WASM computing graph cannot provide the consistency guarantee of its internal operators, but the WASM operator library can truly realize the portability of operators among mobile device, edge and cloud scenarios.

Therefore, in the near future, we plan to release end-to-end integration solutions from the level of WASM operator library (covering the above-mentioned scenarios first), and truly realize the integration of WASM technology into all scenarios in the AI field.

## Acknowledgement

We would like to thank TVM project for providing the compilation stack with WASM backend and TVM community for some precious suggestions about the solution. Also very thanks [@kazum](https://github.com/kazum) for having offered a lot of help when implementing this project.

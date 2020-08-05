---
layout: post
title:  "WASM与TVM在AI领域的结合"
date:   2020-08-03 15:48:01 +0800
categories: machine-learning
---

本文介绍了一种WASM与TVM在AI领域的结合方案：依托TVM端到端深度学习编译全栈的能力，将MindSpore模型编译成WASM字节码，然后在运行时环境中通过Wasmtime进行图加载进而实现模型的无缝迁移部署。

- [背景介绍](#背景介绍)
    - [前期调研](#前期调研)
    - [初步探索](#初步探索)
- [方案介绍](#方案介绍)
    - [方案设计](#方案设计)
    - [代码实现](#代码实现)
    - [原型展示](#原型展示)
- [方案推广](#方案推广)
    - [2020.06 - 内部孵化](#2020.06---内部孵化)
    - [2020.07 - TVM社区推广](#2020.07---tvm社区推广)
- [未来规划](#未来规划)
    - [TVM社区联动](#tvm社区联动)
    - [WASM算子库](#wasm算子库)

## 背景介绍

### 前期调研

[WebAssembly技术](https://webassembly.org/)作为浏览器领域的“新起之秀”，在其他领域也是愈来愈受欢迎。它主打的可移植和安全两大特性，几乎可以应用于各个领域，那么在AI领域WASM又会碰撞出什么样的火花呢？

当前业界针对WASM技术在AI领域已经有了比较多的探索：[TF.js社区](https://blog.tensorflow.org/2020/03/introducing-webassembly-backend-for-tensorflow-js.html)基于WASM编译传统手写算子提升执行速度；[TVM社区](https://tvm.apache.org/2020/05/14/compiling-machine-learning-to-webassembly-and-webgpu)基于WASM编译模型用于浏览器领域的模型推理；还有利用WASM可移植性解决算子库与硬件设备不兼容的问题（详见[XNNPACK](https://github.com/google/XNNPACK)）等等。

### 初步探索

之前我们团队分享了WASM与AI领域结合的初步思路（详见[此处](https://leonwanghui.github.io/machine-learning/2020/04/15/some-thoughts-on-using-wasm-in-ml.html)），正如TF.js和TVM社区开展的探索工作，我们发现WASM具有的可移植性天然地解决了AI模型在全场景落地的问题：针对传统深度学习框架定义的模型，用户在不同硬件环境上进行模型训练/推理时必须要进行额外的定制化开发工作，甚至需要单独开发一套推理引擎系统。

那么如何利用WASM的可移植性实现硬件环境的统一呢？以MindSpore深度学习框架为例，如果我们把MindSpore模型分别从宏观和微观的角度来分析，宏观来看它就是一张基于MindSpore IR定义的**计算图**，微观来看它是一系列MindSpore**算子**的集合。那么我们就可以尝试分别从计算图和算子的维度将WASM与深度学习框架进行结合，也就是提出`WASM计算图`和`WASM算子库`这两种概念。

* WASM计算图

    WASM计算图，顾名思义就是将训练好的模型（包括模型参数）编译成WASM字节码，然后在Runtime环境中通过WASM Runtime加载便可直接进行模型推理，借助WASM的可移植性可以实现任何环境下的模型推理工作：
    * Web领域通过`Emscripten`工具将WASM字节码加载到JS Runtime中进而在浏览器中执行；
    * 非Web领域通过`Wasmtime`工具加载WASM字节码到系统环境中执行。

    对于WASM计算图这种情况，由于训练好的模型（和参数）都是提前保存在系统环境中，因此需要引入`WASI`接口与系统资源进行交互，进而完成离线加载模型的操作。所以在选择WASM Runtime的时候需要选择支持WASI（WASM System Interface）标准的工具（例如Wasmtime），或者也可以像TVM社区那样简单粗暴地直接对Emscripten进行WASI扩展。

* WASM算子库

    WASM算子库相对来说比较好理解，就是把单个算子编译成WASM字节码，然后对上层框架提供一种封装好的算子调用接口。但是和传统手写算子的调用方式不同，框架需要通过一种类似于动态链接的方式来加载WASM算子，但考虑到当前WASM本身不支持动态链接的方式，因此需要提前将所有编译好的WASM算子进行整合，然后对框架层提供算子库的调用接口。

## 方案介绍

通过对上述两种思路进行分析比较，同时在**借鉴了TVM社区已有工作**的情况下，我们决定首先从`WASM计算图`这条路开始进行深入探索，最大程度地利用TVM全栈编译的能力快速实现方案的原型。

**注意：** 如果大家对TVM项目不太了解的话，请移步至[TVM社区](https://tvm.apache.org/)查阅详细资料。

### 方案设计

WASM计算图的设计思路在上一章节已经描述过了，本章节主要展示的是如何利用TVM编译栈来实现上述编译过程。

* WASM图生成
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
    如上图所示，我们可以利用TVM Relay的Python接口直接把模型编译成`graph.o`的可执行文件，但是需要注意的是生成的graph.o文件无法直接被WASM runtime模块识别，必须首先要通过TVM的Rust runtime加载然后通过Rust编译器把图中所示的`WASM Graph Builder`模块直接编译成WASM字节码（即图中的`wasm_graph.wasm`文件）。为什么非得要经过这一步繁琐的转换呢？主要是因为`graph.o`文件中包含了Relay和TVM IR的原语，我们无法直接将这些原语转换成WASM的原语，具体转换的步骤这里就不做赘述了。

* WASM图加载
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
    图加载阶段（由上图看来）似乎是非常简单的，但是实际情况要复杂地多。首先，WASM的运行时针对WASM IR定义了一整套汇编层面的用户接口，这对于上层应用开发者来说是极度不友好的；其次，WASM当前只支持整数类型（例如i32、u64等）作为函数参数，这就导致深度学习领域的张量类型无法通过原生方式传入；更别说还要增加thread、SIMD128这些高级特性的支持等等。

    当然每个新领域的探索都离不开各种各样的问题，而且解决问题本身就是技术/科研人员的本职工作，所以我们没有寄希望于WASM社区而是主动尝试解决这些问题：既然WASM没有面向上层用户的高级API，我们就根据自己的需求开发一套；虽然WASM不支持传入struct或pointer，我们可以通过Memory机制将数据提前写入到WASM内存中然后将内存地址转成i32类型作为函数参数。虽然有些改动有点“反人类”，但是它可以清晰地展示出我们的思路和想法，这就已经足够了。

### 代码实现

由于篇幅有限，这里贴出项目实现的完整代码 (https://github.com/leonwanghui/ms-backend-wasm/tree/master/wasm-standalone-tvm)，欢迎感兴趣的大佬进行交流讨论。

**2020.07更新：** 该方案已经被TVM社区收纳了，详情请见 (https://github.com/apache/incubator-tvm/tree/master/apps/wasm-standalone)。

下面展示一下项目整体的codebase：
```
wasm-standalone/
├── README.md
├── wasm-graph      // WASM图生成模块
│   ├── build.rs    // build脚本
│   ├── Cargo.toml  // 项目依赖包
│   ├── lib         // 通过TVM Relay API编译生成的计算图的存放目录
│   │   ├── graph.json
│   │   ├── graph.o
│   │   ├── graph.params
│   │   └── libgraph_wasm32.a
│   ├── src         // WASM图生成模块源代码
│   │   ├── lib.rs
│   │   ├── types.rs
│   │   └── utils.rs
│   └── tools       // Relay Python API编译脚本的存放目录
│       ├── build_graph_lib.py
└── wasm-runtime    // WASM图生成模块
    ├── Cargo.toml
    ├── src         // WASM图生成模块源代码
    │   ├── graph.rs
    │   ├── lib.rs
    │   └── types.rs
    └── tests      // WASM图生成模块测试用例
        └── test_graph_resnet50
```

### 原型展示

为了让大家对该方案有一个更形象具体的理解，我们准备了一个简单的原型：通过TVM Relay API将基于ONNX生成的ResNet50模型编译成`wasm_graph_resnet50.wasm`文件，然后在运行时环境中通过Wasmtime加载WASM完成模型推理功能（具体操作流程详见[此处](https://github.com/apache/incubator-tvm/blob/master/apps/wasm-standalone/README.md#poc-guidelines)）。

## 方案推广

作为一个在开源社区耕耘了长达四年的开发团队，我们早已养成了四个Open（Open Source, Open Design, Open Development以及Open Community）的习惯，本章节就带大家一块回顾下我们在推广WASM与AI领域结合方案的心路历程。

### 2020.06 - 内部孵化

对于开源界一直流传的一句箴言：“Talk is cheap, show me the code”，对于方案推广阶段来说必须要有些拿出讲的干货。因此我们花了不到一个月时间完成了从构思到开发出方案原型的工作，这为我们后期在TVM社区推广打下了坚实的基础。

### 2020.07 - TVM社区推广

由于该方案是基于TVM编译栈构建的，因此在方案完成后我们的第一想法就是把它全部回馈给TVM社区，同时自荐作为owner进行后期的特性开发和维护。经过这一个月来在社区的持续推动，最初提出的方案也进行了好几轮大的重构整改，终于在7月底被社区采纳合入，感兴趣的话可以点击社区PR[#5892](https://github.com/apache/incubator-tvm/pull/5892)查阅详情。

## 未来规划

### TVM社区联动

正如前面所说的，该方案仍处于**试验**阶段，因此我们会和TVM社区一起共同探索更多可能性，目前初步规划的特性有：
* 支持基于SIMD128的数据并行处理；
* 进一步完善TVM社区的Rust runtime API模块，使其能原生支持WASM Memory特性的对接；
* 基于WASM后端的AutoTVM优化；
* 更多网络支持。

### WASM算子库

当前我们只是针对WASM计算图这个方向进行了深入探索，但如果要是将WASM技术与深度学习框架（比如MindSpore）相结合的话，WASM算子库的方向可能会释放更大的潜能。这里首先列举几个更适合WASM算子库的场景：
* 很多深度学习框架本身已经定义了自己的IR以及编译流水线，只有WASM算子库可以无缝地与这些框架的图编译层相融合；
* WASM计算图只能用于模型推理，而WASM算子库可以适用于模型训练/验证/推理等场景；
* 在可移植性这个层面，WASM计算图无法提供其内部算子的一致性保证，而WASM算子库真正实现了端边云全场景中算子的可移植性。

因此，我们近期同样会从WASM算子库这个层面梳理出一套端到到的集成方案（优先覆盖上述几个场景），真正实现WASM技术在AI领域全场景的结合。

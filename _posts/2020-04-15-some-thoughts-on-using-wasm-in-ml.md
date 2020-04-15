---
layout: post
title:  "ML与WASM的结合与思考"
date:   2020-04-15 17:03:43 +0800
categories: machine-learning
---

最近读了一篇博客（[Introducing the WebAssembly backend for TensorFlow.js](https://blog.tensorflow.org/2020/03/introducing-webassembly-backend-for-tensorflow-js.html)）感触颇深，今天想跟大家谈谈WebAssembly（后文简称WASM）与ML领域的结合点。

## 术语

为避免大家在阅读后文过程中产生疑惑，这里首先针对部分术语进行概括阐述：
- **WebAssembly**（又称WASM）：基于二进制操作指令的栈式结构的虚拟机，可以被编译为机器码，进而更快、高效地执行本地方法和硬件资源；
- **WASM System Interface**（又称WASI）：针对非Web领域制定的一套调用逻辑操作系统的标准接口，使得WASM平台可在不同操作系统间无缝迁移；
- **TensorFlow.js**：是一个 JavaScript 库，用于在浏览器和 Node.js 上训练和部署模型；
- **TensorFlow Lite**：用于在移动设备和嵌入式设备上部署模型的精简库；
- **MindSpore**：面向端边云全场景设计的开源深度学习训练/推理框架；
- **XNNPack**：针对ARM、WebAssembly和x86平台优化的浮点神经网络推理算子库；
- **Emscripten**：基于LLVM将C/C++代码转化成WASM格式的编译器。

## 背景介绍

### 博客内容

3月11日TF社区正式发布了TF.js对WASM的支持，WASM后端作为WebGL的替选方案可以让用户直接在CPU环境上进行模型推理。据原文称，WASM后端很好地结合了性能和可移植的优点：执行速度比原生JS快（2-10倍），同时对终端设备的支持能力比WebGL更好；同时考虑到WASM社区正在开发的SIMD特性，WASM后端的推理性能会进一步提升（2倍以上）。

### WASM发展

![WASM Introduction](/assets/pics/wasm-introduction.png)

通过上段描述大家应该就能猜到WASM的特点：执行速度快和可移植性强。WASM最开始提出主要是为了解决JS代码执行的性能问题，同时由于浏览器场景的限制WASM天然支持具有可移植性的特点，而这两大特点也让它成为了Web领域的运行时标准。

## WASM与ML领域的结合

本章节我将为大家展示WASM与ML领域的三个结合点，并通过展示业界已有案例或规划方案的方式来阐述其可行性。

### 通过WASM静态编译提升模型推理性能

![TF.js WASM Backend](/assets/pics/tfjs-wasm-backend.png)

前文提到的WASM backend与TF.js结合的方案要解决的问题之一就是如何解决TF.js中传统JS算子执行速度过慢。其解决方案也简单粗暴，通过WASM的可移植性将C/C++实现的算子转换换成wasm module然后暴露给TF.js框架供用户调用，这样就能大幅提升算子的执行速度。

![Inference Speed of WebGL vs WASM](/assets/pics/inference-speed-of-webgl-vs-wasm.png)

但是通过WASM和WebGL的性能对比，不难看出WASM的性能提升有限，如果大家看过TF Dev Summit’20的议题 (https://www.youtube.com/watch?v=iH9CS-QYmZs) 就知道当前WASM backend算子执行的速度和Python算子执行的速度差不多。由此可以推断该方案更多的是除WebGL外提出一种新的方向，但是用于生产环境的可行性仍需评估。

### 利用WASM可移植性解决算子库与硬件设备不兼容的问题

如果大家看了`tfjs-backend-js` (https://github.com/tensorflow/tfjs/tree/master/tfjs-backend-wasm) 的源码，就会发现它调用的都是基于`XNNPACK`库 (https://github.com/google/XNNPACK) 封装好的算子。打开XNNPACK之后发现它就是一个支持WASM的通用算子库，那么有哪些框架已经支持了呢？下面有意思的事情出现了：TF.js和TF Lite都支持XNNPACK，这就说明这两个框架可共用一套算子库，这是否就意味着TF通过XNNPACK和WASM覆盖了Web和终端设备等所有场景的模型推理呢？如果是这样的话，开发者可以将TF训练出的模型在Web侧和端侧无缝迁移，解决了模型推理框架与硬件设备不兼容的问题。

### WASM/WASI带来的AI新特性

我们在最近开源了MindSpore (https://www.mindspore.cn/) ，一个新的全场景深度学习训练推理框架。我们看到基于WASM/WASI对跨平台可移植性的支持，是有可能开发一套面向全场景的通用后端算子库。不同于目前仍然需要CPU活着WebGL对GPU算子实现的支持，来使得类似tf wasm这样的方案能够被使用，基于WASM/WASI的通用后端算子库，以及框架本身的WASM Build，在未来有可能会让用户真正体验到Webassembly带来的便利-端边云的后端无感知、安全性和高性能。

WASM也会给像联邦学习这样的新技术带来新的可能，不同于现有的绝大多数面向容器化的部署方式，借助WASI接口的逐渐繁荣，WASM部署可以带来非常好的安全隔离性体验，以及更小的内存开销，让联邦学习的多方计算可以更加安全、高效的进行

![MindSpore WASI Architecture](/assets/pics/mindspore-wasi-architecture.png)

如果您也对这个方向感兴趣，欢迎联系我（<wanghui71leon@gmail.com>），在未来可以一起通过社区进行讨论贡献。

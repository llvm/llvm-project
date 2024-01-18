# 'ArmSME' Dialect

Basic dialect to target Arm SME.

This dialect defines custom and LLVM IR intrinsic operations that are used to
target Arm Scalable Matrix Extension. Through the available conversion and
ArmSME passes you can, for example, lower a
[linalg.matmul](https://mlir.llvm.org/docs/Dialects/Linalg/#linalgmatmul-linalgmatmulop)
opereation to Arm SME
[FMOPA](https://developer.arm.com/documentation/ddi0602/2023-03/SME-Instructions/FMOPA--widening---Half-precision-floating-point-sum-of-outer-products-and-accumulate-)
(floating-point outer product) operations. See one of the in-tree end-to-end
integration tests for reference:

* [Linalg/CPU/ArmSME/matmul.mlir](https://github.com/llvm/llvm-project/blob/main/mlir/test/Integration/Dialect/Linalg/CPU/ArmSME/matmul.mlir)
* [Vector/CPU/ArmSME/test-outerproduct-f64.mlir](https://github.com/llvm/llvm-project/blob/main/mlir/test/Integration/Dialect/Vector/CPU/ArmSME/test-outerproduct-f64.mlir)

These tests are run "post-commit" by the
[clang-aarch64-sve-vla](https://lab.llvm.org/buildbot/#/builders/197) LLVM
BuildBot worker.

**References:**

* [The Scalable Matrix Extension (SME), for Armv9-A](https://developer.arm.com/documentation/ddi0616)
* [A64 -- SME Instructions (alphabetic order)](https://developer.arm.com/documentation/ddi0602/2023-03/SME-Instructions)

[TOC]

## Operations

[include "Dialects/ArmSMEOps.md"]

## Operations for LLVM IR Intrinsics

[include "Dialects/ArmSMEIntrinsicOps.md"]

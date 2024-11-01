//===-- Passes.td - GPU pass definition file ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GPU_PASSES
#define MLIR_DIALECT_GPU_PASSES

include "mlir/Pass/PassBase.td"

def GpuLaunchSinkIndexComputations : Pass<"gpu-launch-sink-index-computations"> {
  let summary = "Sink index computations into gpu.launch body";
  let constructor = "mlir::createGpuLauchSinkIndexComputationsPass()";
  let dependentDialects = ["mlir::gpu::GPUDialect"];
}

def GpuKernelOutlining : Pass<"gpu-kernel-outlining", "ModuleOp"> {
  let summary = "Outline gpu.launch bodies to kernel functions";
  let constructor = "mlir::createGpuKernelOutliningPass()";
  let dependentDialects = ["mlir::DLTIDialect"];
}

def GpuAsyncRegionPass : Pass<"gpu-async-region", "func::FuncOp"> {
  let summary = "Make GPU ops async";
  let constructor = "mlir::createGpuAsyncRegionPass()";
  let dependentDialects = ["async::AsyncDialect"];
}

def GpuMapParallelLoopsPass
    : Pass<"gpu-map-parallel-loops", "mlir::func::FuncOp"> {
  let summary = "Greedily maps loops to GPU hardware dimensions.";
  let constructor = "mlir::createGpuMapParallelLoopsPass()";
  let description = "Greedily maps loops to GPU hardware dimensions.";
  let dependentDialects = ["mlir::gpu::GPUDialect"];
}

def GPULowerMemorySpaceAttributesPass
    : Pass<"gpu-lower-memory-space-attributes"> {
  let summary = "Assign numeric values to memref memory space symbolic placeholders";
  let description = [{
    Updates all memref types that have a memory space attribute
    that is a `gpu::AddressSpaceAttr`. These attributes are
    changed to `IntegerAttr`'s using a mapping that is given in the
    options.
  }];
  let options = [
    Option<"privateAddrSpace", "private", "unsigned", "5",
      "private address space numeric value">,
    Option<"workgroupAddrSpace", "workgroup", "unsigned", "3",
      "workgroup address space numeric value">,
    Option<"globalAddrSpace", "global", "unsigned", "1",
      "global address space numeric value">
  ];
}

#endif // MLIR_DIALECT_GPU_PASSES

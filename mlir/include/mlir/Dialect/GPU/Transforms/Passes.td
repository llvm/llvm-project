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

def GpuLaunchSinkIndexComputationsPass
    : Pass<"gpu-launch-sink-index-computations"> {
  let summary = "Sink index computations into gpu.launch body";
}

def GpuKernelOutliningPass : Pass<"gpu-kernel-outlining", "ModuleOp"> {
  let summary = "Outline gpu.launch bodies to kernel functions";
  let dependentDialects = ["mlir::DLTIDialect", "cf::ControlFlowDialect"];
  let options = [Option<"dataLayoutStr", "data-layout-str", "std::string",
                        /*default=*/"",
                        "String description of the data layout">];
}

def GpuAsyncRegionPass : Pass<"gpu-async-region", "func::FuncOp"> {
  let summary = "Make GPU ops async";
  let dependentDialects = ["async::AsyncDialect"];
}

def GpuMapParallelLoopsPass
    : Pass<"gpu-map-parallel-loops", "mlir::func::FuncOp"> {
  let summary = "Greedily maps loops to GPU hardware dimensions.";
  let description = [{
    Maps the parallel loops found in the given function to workgroups. The first
    loop encountered will be mapped to the global workgroup and the second loop
    encountered to the local workgroup. Within each mapping, the first three
    dimensions are mapped to x/y/z hardware ids and all following dimensions are
    mapped to sequential loops.

    Ordering of the loop mapping against the different dimensions is controlled
    by the `mapping-policy` option.
    Two policies are supported:
       1. `outermost-first` (default): the outermost loop maps to X, then Y
          and finally Z.
       2. `innermost-first`: the innermost loop maps to X, then Y and finally Z.
  }];
  let dependentDialects = ["mlir::gpu::GPUDialect"];
  let options = [Option<"mappingPolicyStr", "mapping-policy", "std::string",
                        /*default=*/"\"outermost-first\"",
                        "Policy outlining how to assign loops to GPU dimensions."
                        "Supported values are `outermost-first` and `innermost-first`.">];
}

def GpuEliminateBarriers
    : Pass<"gpu-eliminate-barriers", "mlir::func::FuncOp"> {
  let summary = "Erase unnecessary barriers";
  let description = [{
    Barrier elimination pass. If a barrier does not enforce any conflicting
    pair of memory effects, including a pair that is enforced by another
    barrier, it is unnecessary and can be removed. Adapted from
    "High-Performance GPU-to-CPU Transpilation and Optimization via High-Level
    Parallel Constructs" by Moses, Ivanov, Domke, Endo, Doerfert, and Zinenko in
    PPoPP 2023 and implementation in Polygeist.
  }];
  let dependentDialects = [
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect"
  ];
}

def GpuDecomposeMemrefsPass : Pass<"gpu-decompose-memrefs"> {
  let summary = "Decomposes memref index computation into explicit ops.";
  let description = [{
    This pass decomposes memref index computation into explicit computations on
    sizes/strides, obtained from `memref.extract_memref_metadata` which it tries
    to place outside of `gpu.launch` body. Memrefs are then reconstructed using
    `memref.reinterpret_cast`.
    This is needed for as some targets (SPIR-V) lower memrefs to bare pointers
    and sizes/strides for dynamically-sized memrefs are not available inside
    `gpu.launch`.
  }];
  let dependentDialects = [
    "mlir::memref::MemRefDialect",
    "mlir::affine::AffineDialect"
  ];
}

def GpuModuleToBinaryPass
    : Pass<"gpu-module-to-binary", ""> {
  let summary = "Transforms a GPU module into a GPU binary.";
  let description = [{
    This pass searches for all nested GPU modules and serializes the module
    using the target attributes attached to the module, producing a GPU binary
    with an object for every target.

    The `format` argument can have the following values:
    1. `offloading`, `llvm`: produces an offloading representation.
    2. `assembly`, `isa`: produces assembly code.
    3. `binary`, `bin`: produces binaries.
    4. `fatbinary`, `fatbin`: produces fatbinaries.
  }];
  let options = [
    Option<"toolkitPath", "toolkit", "std::string", [{""}],
           "Toolkit path.">,
    ListOption<"linkFiles", "l", "std::string",
           "Extra files to link to.">,
    Option<"cmdOptions", "opts", "std::string", [{""}],
           "Command line options to pass to the tools.">,
    Option<"compilationTarget", "format", "std::string", [{"fatbin"}],
           "The target representation of the compilation process.">,
    Option<"elfSection", "section", "std::string", [{""}],
           "ELF section where binary is to be located.">
  ];
}

#endif // MLIR_DIALECT_GPU_PASSES

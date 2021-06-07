//===-- Passes.td - Sparse tensor pass definition file -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_PASSES
#define MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def Sparsification : Pass<"sparsification", "ModuleOp"> {
  let summary = "Automatically generate sparse tensor code from annotations";
  let constructor = "mlir::createSparsificationPass()";
  let dependentDialects = [
    "LLVM::LLVMDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "sparse_tensor::SparseTensorDialect",
    "vector::VectorDialect",
  ];
  let options = [
    Option<"parallelization", "parallelization-strategy", "int32_t", "0",
           "Set the parallelization strategy">,
    Option<"vectorization", "vectorization-strategy", "int32_t", "0",
           "Set the vectorization strategy">,
    Option<"vectorLength", "vl", "int32_t", "1",
           "Set the vector length">,
    Option<"enableSIMDIndex32", "enable-simd-index32", "bool", "false",
           "Enable i32 indexing into vectors (for efficiency)">
  ];
}

def SparseTensorConversion : Pass<"sparse-tensor-conversion", "ModuleOp"> {
  let summary = "Apply conversion rules to sparse tensors";
  let constructor = "mlir::createSparseTensorConversionPass()";
  let dependentDialects = [
    "LLVM::LLVMDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "sparse_tensor::SparseTensorDialect",
    "vector::VectorDialect",
  ];
}

#endif // MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_PASSES


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
  let summary = "Automatically generate sparse tensor code from sparse tensor types";
  let description = [{
    A pass that implements the core functionality of a **sparse compiler**.
    Each Linalg operation (MLIR's tensor index notation) that operates on
    sparse tensor types is converted into code in which the sparsity is
    explicit both in terms of co-iterating looping logic as well as
    selected sparse storage schemes.

    See the `SparseTensor` dialect documentation for more background.

    Example input:

    ```mlir
    #matvec = {
      indexing_maps = [
        affine_map<(i,j) -> (i,j)>, // A
        affine_map<(i,j) -> (j)>,   // b
        affine_map<(i,j) -> (i)>    // x (out)
      ],
      iterator_types = ["parallel", "reduction"],
      doc = "X(i) += A(i,j) * B(j)"
    }

    // Multiply a sparse matrix A with a dense vector b into a dense vector x.
    func @kernel_matvec(%arga: tensor<?x?xf64, #SparseMatrix>,
                        %argb: tensor<?xf64>,
                        %argx: tensor<?xf64>) -> tensor<?xf64> {
      %0 = linalg.generic #matvec
        ins(%arga, %argb: tensor<?x?xf64, #SparseMatrix>, tensor<?xf64>)
        outs(%argx: tensor<?xf64>) {
        ^bb(%a: f64, %b: f64, %x: f64):
          %0 = mulf %a, %b : f64
          %1 = addf %x, %0 : f64
          linalg.yield %1 : f64
      } -> tensor<?xf64>
      return %0 : tensor<?xf64>
    }
    ```
  }];
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
  let summary = "Apply conversion rules to sparse tensor primitives and types";
  let description = [{
    A pass that converts sparse tensor primitives to calls into a runtime
    support library. All sparse tensor types are converted into opaque
    pointers to the underlying sparse storage schemes.

    Note that this is a current implementation choice to keep the conversion
    relatively simple. In principle, these primitives could also be
    converted to actual elaborate IR code that implements the primitives
    on the selected sparse tensor storage schemes.

    Example of the conversion:

    ```mlir
      Before:
        %c1 = constant 1 : index
        %0 = sparse_tensor.pointers %arg0, %c1
          : tensor<8x8xf32, #sparse_tensor.encoding<{
              dimLevelType = [ "dense", "compressed" ],
              pointerBitWidth = 0,
              indexBitWidth = 0
            }>> to memref<?xindex>

      After:
        %c1 = constant 1 : index
        %0 = call @sparsePointers(%arg0, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
    ```
  }];
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

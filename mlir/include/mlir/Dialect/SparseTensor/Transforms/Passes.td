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

def PreSparsificationRewrite : Pass<"pre-sparsification-rewrite", "ModuleOp"> {
  let summary = "Applies sparse tensor rewriting rules prior to sparsification";
  let description = [{
    A pass that applies rewriting rules to sparse tensor operations prior
    to running the actual sparsification pass.
  }];
  let constructor = "mlir::createPreSparsificationRewritePass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "bufferization::BufferizationDialect",
    "linalg::LinalgDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "sparse_tensor::SparseTensorDialect",
  ];
}

def SparsificationPass : Pass<"sparsification", "ModuleOp"> {
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
    func.func @kernel_matvec(%arga: tensor<?x?xf64, #SparseMatrix>,
                             %argb: tensor<?xf64>,
                             %argx: tensor<?xf64>) -> tensor<?xf64> {
      %0 = linalg.generic #matvec
        ins(%arga, %argb: tensor<?x?xf64, #SparseMatrix>, tensor<?xf64>)
        outs(%argx: tensor<?xf64>) {
        ^bb(%a: f64, %b: f64, %x: f64):
          %0 = arith.mulf %a, %b : f64
          %1 = arith.addf %x, %0 : f64
          linalg.yield %1 : f64
      } -> tensor<?xf64>
      return %0 : tensor<?xf64>
    }
    ```
  }];
  let constructor = "mlir::createSparsificationPass()";
  let dependentDialects = [
    "AffineDialect",
    "arith::ArithDialect",
    "bufferization::BufferizationDialect",
    "LLVM::LLVMDialect",
    "linalg::LinalgDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "sparse_tensor::SparseTensorDialect",
  ];
  // TODO(57514): These enum options are duplicated in Passes.h.
  let options = [
    Option<"enableIndexReduction", "enable-index-reduction", "bool",
           "false",
           "Enable dependent index reduction based algorithm to handle non-trivial index expressions on sparse inputs (experimental features)">,
    Option<"parallelization", "parallelization-strategy", "mlir::SparseParallelizationStrategy",
           "mlir::SparseParallelizationStrategy::kNone",
           "Set the parallelization strategy", [{llvm::cl::values(
             clEnumValN(mlir::SparseParallelizationStrategy::kNone, "none",
                        "Turn off sparse parallelization."),
             clEnumValN(mlir::SparseParallelizationStrategy::kDenseOuterLoop,
                        "dense-outer-loop",
                        "Enable dense outer loop sparse parallelization."),
             clEnumValN(mlir::SparseParallelizationStrategy::kAnyStorageOuterLoop,
                        "any-storage-outer-loop",
                        "Enable sparse parallelization regardless of storage for the outer loop."),
             clEnumValN(mlir::SparseParallelizationStrategy::kDenseAnyLoop,
                        "dense-any-loop",
                        "Enable dense parallelization for any loop."),
             clEnumValN(mlir::SparseParallelizationStrategy::kAnyStorageAnyLoop,
                        "any-storage-any-loop",
                        "Enable sparse parallelization for any storage and loop."))}]>
  ];
}

def PostSparsificationRewrite : Pass<"post-sparsification-rewrite", "ModuleOp"> {
  let summary = "Applies sparse tensor rewriting rules after sparsification";
  let description = [{
    A pass that applies rewriting rules to sparse tensor operations after
    running the actual sparsification pass.
  }];
  let constructor = "mlir::createPostSparsificationRewritePass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "bufferization::BufferizationDialect",
    "linalg::LinalgDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "sparse_tensor::SparseTensorDialect",
  ];
  let options = [
    Option<"enableRuntimeLibrary", "enable-runtime-library", "bool",
           "true", "Enable runtime library for manipulating sparse tensors">,
    Option<"enableForeach", "enable-foreach", "bool",
           "true", "Enable rewriting rules for the foreach operator">,
    Option<"enableConvert", "enable-convert", "bool",
           "true", "Enable rewriting rules for the convert operator">,
  ];
}

def SparseTensorConversionPass : Pass<"sparse-tensor-conversion", "ModuleOp"> {
  let summary = "Convert sparse tensors and primitives to library calls";
  let description = [{
    A pass that converts sparse tensor primitives into calls into a runtime
    support library. Sparse tensor types are converted into opaque pointers
    to the underlying sparse storage schemes.

    The use of opaque pointers together with runtime support library keeps
    the conversion relatively simple, but at the expense of IR opacity,
    which obscures opportunities for subsequent optimization of the IR.
    An alternative is provided by the SparseTensorCodegen pass.

    Example of the conversion:

    ```mlir
      Before:
        func.func @foo(%arg0: tensor<8x8xf32, #CSR>) -> memref<?xindex> {
          %0 = sparse_tensor.pointers %arg0 {dimension = 1 : index}
             : tensor<8x8xf32, #CSR> to memref<?xindex>
          return %0 : memref<?xindex>
        }

      After:
        func.func @foo(%arg0: !llvm.ptr<i8>) -> memref<?xindex> {
          %c1 = arith.constant 1 : index
          %0 = call @sparsePointers0(%arg0, %c1)
             : (!llvm.ptr<i8>, index) -> memref<?xindex>
          return %0 : memref<?xindex>
        }
    ```
  }];
  let constructor = "mlir::createSparseTensorConversionPass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "bufferization::BufferizationDialect",
    "LLVM::LLVMDialect",
    "linalg::LinalgDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "sparse_tensor::SparseTensorDialect",
  ];
  let options = [
    Option<"sparseToSparse", "s2s-strategy", "int32_t", "0",
           "Set the strategy for sparse-to-sparse conversion">,
  ];
}

def SparseTensorCodegen : Pass<"sparse-tensor-codegen", "ModuleOp"> {
  let summary = "Convert sparse tensors and primitives to actual code";
  let description = [{
    A pass that converts sparse tensor types and primitives to actual
    compiler visible buffers and compiler IR that implements these
    primitives on the selected sparse tensor storage schemes.

    This pass provides an alternative to the SparseTensorConversion pass,
    eliminating the dependence on a runtime support library, and providing
    much more opportunities for subsequent compiler optimization of the
    generated code.

    Example of the conversion:

    ```mlir
      Before:
        func.func @foo(%arg0: tensor<8x8xf32, #CSR>) -> memref<?xindex> {
          %0 = sparse_tensor.pointers %arg0 {dimension = 1 : index}
             : tensor<8x8xf32, #CSR> to memref<?xindex>
          return %0 : memref<?xindex>
        }

      After:
        func.func @foo(%arg0: memref<2xindex>,
                       %arg1: memref<3xindex>,
                       %arg2: memref<?xindex>,
                       %arg3: memref<?xindex>,
                       %arg4: memref<?xf32>) -> memref<?xindex> {
          return %arg2 : memref<?xindex>
        }
    ```
  }];
  let constructor = "mlir::createSparseTensorCodegenPass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "bufferization::BufferizationDialect",
    "linalg::LinalgDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "sparse_tensor::SparseTensorDialect",
  ];
  let options = [
    Option<"enableBufferInitialization", "enable-buffer-initialization", "bool",
           "false", "Enable zero-initialization of the memory buffers">,
    Option<"createSparseDeallocs", "create-sparse-deallocs", "bool",
           "true", "Specify if the temporary buffers created by the sparse "
                   "compiler should be deallocated. For compatibility with core "
                   "bufferization passes. "
                   "This option is only used when enable-runtime-library=false. "
                   "See also create-deallocs for BufferizationOption.">,
  ];
}

def SparseBufferRewrite : Pass<"sparse-buffer-rewrite", "ModuleOp"> {
  let summary = "Rewrite sparse primitives on buffers to actual code";
  let description = [{
    A pass that rewrites sparse primitives on buffers to the MLIR implementation
    of the primitives. For example, sparse_tensor.sort operator is implemented
    in this pass.
  }];
  let constructor = "mlir::createSparseBufferRewritePass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "linalg::LinalgDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "sparse_tensor::SparseTensorDialect",
  ];
  let options = [
    Option<"enableBufferInitialization", "enable-buffer-initialization", "bool",
           "false", "Enable zero-initialization of the memory buffers">,
  ];
}

def SparseVectorization : Pass<"sparse-vectorization", "ModuleOp"> {
  let summary = "Vectorizes loops after sparsification";
  let description = [{
    A pass that converts loops after sparsification into vector loops.
    The vector dialect is used as target to provide an architectural
    neutral way of exploiting any platform that supports SIMD instructions.

    The vector length (viz. `vl`) describes the number of packed data elements
    (e.g. both vector<16xf32> and vector<16xf64> have a vector length of 16 even
    though the actual bitwidths differ). A small multiple of the actual lengths
    supported in hardware typically results in efficient SIMD code, since the
    backend will map longer vectors to multiple vector registers, thereby
    effectively unrolling an addition level within the generated for-loop.

    Example of the conversion:

    ```mlir
      Before:
        %3 = memref.load %2[] : memref<f32>
        %4 = scf.for %arg3 = %c0 to %c1024 step %c1 iter_args(%arg4 = %3) -> (f32) {
          %6 = memref.load %0[%arg3] : memref<?xf32>
          %7 = memref.load %1[%arg3] : memref<1024xf32>
          %8 = arith.mulf %6, %7 : f32
          %9 = arith.addf %arg4, %8 : f32
          scf.yield %9 : f32
        }
        memref.store %4, %2[] : memref<f32>

      After:
        %3 = memref.load %2[] : memref<f32>
        %4 = vector.insertelement %3, %cst[%c0 : index] : vector<32xf32>
        %5 = scf.for %arg3 = %c0 to %c1024 step %c32 iter_args(%arg4 = %4) -> (vector<32xf32>) {
          %8 = vector.load %0[%arg3] : memref<?xf32>, vector<32xf32>
          %9 = vector.load %1[%arg3] : memref<1024xf32>, vector<32xf32>
          %10 = arith.mulf %8, %9 : vector<32xf32>
          %11 = arith.addf %arg4, %10 : vector<32xf32>
          scf.yield %11 : vector<32xf32>
        }
        %6 = vector.reduction <add>, %5 : vector<32xf32> into f32
        memref.store %6, %2[] : memref<f32>
    ```
  }];
  let constructor = "mlir::createSparseVectorizationPass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "sparse_tensor::SparseTensorDialect",
    "vector::VectorDialect",
  ];
  let options = [
    Option<"vectorLength", "vl", "int32_t", "0",
           "Set the vector length (use 0 to disable vectorization)">,
    Option<"enableVLAVectorization", "enable-vla-vectorization", "bool",
           "false", "Enable vector length agnostic vectorization">,
    Option<"enableSIMDIndex32", "enable-simd-index32", "bool", "false",
           "Enable i32 indexing into vectors (for efficient gather/scatter)">,
  ];
}

def StorageSpecifierToLLVM : Pass<"sparse-storage-specifier-to-llvm", "ModuleOp"> {
  let summary = "Lower sparse storage specifer to llvm structure";
  let description = [{
     This pass rewrites sparse tensor storage specifier-related operations into
     LLVMDialect, and converts sparse tensor storage specifier into an llvm.struct.

     Example of the conversion:
     ```mlir
     Before:
       %0 = sparse_tensor.storage_specifier.get %arg0 dim_sz at 0
       : !sparse_tensor.storage_specifier<#CSR> to i64

     After:
       %0 = llvm.extractvalue %arg0[0, 0] : !llvm.struct<(array<2 x i64>, array<3 x i64>)>
     ```
  }];
  let constructor = "mlir::createStorageSpecifierToLLVMPass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "LLVM::LLVMDialect",
    "sparse_tensor::SparseTensorDialect",
  ];
}

#endif // MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_PASSES

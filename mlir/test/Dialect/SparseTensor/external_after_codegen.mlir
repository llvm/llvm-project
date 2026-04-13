// RUN: mlir-opt %s --sparse-tensor-codegen --sparse-assembler | FileCheck %s

// Regression test for https://github.com/llvm/llvm-project/issues/183776:
// Running --sparse-assembler after --sparse-tensor-codegen must not crash.
// After codegen, sparse tensor arguments are replaced by memrefs and
// \!sparse_tensor.storage_specifier types. getSparseTensorEncoding() returns
// non-null for StorageSpecifierType, but convTypes()/convVals() must not
// attempt cast<RankedTensorType> on it. Instead, non-RankedTensorType types
// with a sparse encoding should pass through unchanged.

#CSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0: dense, d1: compressed) }>

// Storage_specifier types from codegen must pass through sparse-assembler
// unchanged (not be treated as sparse tensor arguments to wrap).
// CHECK-LABEL: func.func @storage_specifier_passthrough(
// CHECK-SAME:    storage_specifier
// CHECK-SAME:    storage_specifier
// CHECK:         return %{{.*}} : tensor<32x32xf32>
func.func @storage_specifier_passthrough(%arg0: tensor<32x32xf32, #CSR>,
                                         %arg1: tensor<32x32xf32, #CSR>)
    -> tensor<32x32xf32> {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<32x32xf32>
  %out = linalg.fill ins(%cst : f32) outs(%init : tensor<32x32xf32>)
      -> tensor<32x32xf32>
  %3 = linalg.add
      ins(%arg0, %arg1 : tensor<32x32xf32, #CSR>, tensor<32x32xf32, #CSR>)
      outs(%out : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %3 : tensor<32x32xf32>
}

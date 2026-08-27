// RUN: mlir-opt -pass-pipeline="builtin.module(func.func(convert-arith-to-llvm{index-bitwidth=32}))" %s | FileCheck %s --check-prefix=CHECK32
// RUN: mlir-opt -pass-pipeline="builtin.module(func.func(convert-arith-to-llvm{index-bitwidth=64}))" %s | FileCheck %s --check-prefix=CHECK64
// RUN: mlir-opt -pass-pipeline="builtin.module(func.func(convert-arith-to-llvm{index-bitwidth=128}))" %s | FileCheck %s --check-prefix=CHECK128

// An `index` constant is retyped to the converted index type, which is signless
// but holds signed values. Narrowing therefore truncates and widening
// sign-extends. 

// CHECK32-LABEL: @scalar_index_constant
//       CHECK32:   llvm.mlir.constant(-1 : i32) : i32
//       CHECK32:   llvm.mlir.constant(0 : i32) : i32
// CHECK64-LABEL: @scalar_index_constant
//       CHECK64:   llvm.mlir.constant(-1 : i64) : i64
//       CHECK64:   llvm.mlir.constant(4294967296 : i64) : i64
// CHECK128-LABEL: @scalar_index_constant
//       CHECK128:   llvm.mlir.constant(-1 : i128) : i128
//       CHECK128:   llvm.mlir.constant(4294967296 : i128) : i128
func.func @scalar_index_constant() -> (index, index) {
  %0 = arith.constant -1 : index
  %1 = arith.constant 4294967296 : index
  return %0, %1 : index, index
}

// CHECK32-LABEL: @dense_index_constant
//       CHECK32:   llvm.mlir.constant(dense<[-1, 0]> : vector<2xi32>) : vector<2xi32>
// CHECK64-LABEL: @dense_index_constant
//       CHECK64:   llvm.mlir.constant(dense<[-1, 4294967296]> : vector<2xi64>) : vector<2xi64>
// CHECK128-LABEL: @dense_index_constant
//       CHECK128:   llvm.mlir.constant(dense<[-1, 4294967296]> : vector<2xi128>) : vector<2xi128>
func.func @dense_index_constant() -> vector<2xindex> {
  %0 = arith.constant dense<[-1, 4294967296]> : vector<2xindex>
  return %0 : vector<2xindex>
}

// CHECK32-LABEL: @sparse_index_constant
//       CHECK32:   llvm.mlir.constant(sparse<0, -1> : vector<4xi32>) : vector<4xi32>
// CHECK64-LABEL: @sparse_index_constant
//       CHECK64:   llvm.mlir.constant(sparse<0, -1> : vector<4xi64>) : vector<4xi64>
// CHECK128-LABEL: @sparse_index_constant
//       CHECK128:   llvm.mlir.constant(sparse<0, -1> : vector<4xi128>) : vector<4xi128>
func.func @sparse_index_constant() -> vector<4xindex> {
  %0 = arith.constant sparse<[[0]], [-1]> : vector<4xindex>
  return %0 : vector<4xindex>
}

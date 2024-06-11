// RUN: mlir-opt %s -sparse-storage-specifier-to-llvm --cse --canonicalize | FileCheck %s

#CSR = #sparse_tensor.encoding<{map = (d0, d1) -> (d0 : dense, d1 : compressed)}>

// CHECK-LABEL:   func.func @sparse_metadata_init() -> !llvm.struct<(array<2 x i64>, array<3 x i64>)> {
// CHECK-DAG:       %[[STRUCT:.*]] = llvm.mlir.undef : !llvm.struct<(array<2 x i64>, array<3 x i64>)>
// CHECK-DAG:       %[[CST0:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_1:.*]] = llvm.insertvalue %[[CST0]], %[[STRUCT]][1, 0] : !llvm.struct<(array<2 x i64>, array<3 x i64>)>
// CHECK:           %[[VAL_2:.*]] = llvm.insertvalue %[[CST0]], %[[VAL_1]][1, 1] : !llvm.struct<(array<2 x i64>, array<3 x i64>)>
// CHECK:           %[[VAL_3:.*]] = llvm.insertvalue %[[CST0]], %[[VAL_2]][1, 2] : !llvm.struct<(array<2 x i64>, array<3 x i64>)>
// CHECK:           return %[[VAL_3]] : !llvm.struct<(array<2 x i64>, array<3 x i64>)>
// CHECK:         }
func.func @sparse_metadata_init() -> !sparse_tensor.storage_specifier<#CSR> {
  %0 = sparse_tensor.storage_specifier.init : !sparse_tensor.storage_specifier<#CSR>
  return %0 : !sparse_tensor.storage_specifier<#CSR>
}

// CHECK-LABEL:   func.func @sparse_get_md(
// CHECK-SAME:      %[[VAL_0:.*]]: !llvm.struct<(array<2 x i64>, array<3 x i64>)>) -> index {
// CHECK:           %[[VAL_1:.*]] = llvm.extractvalue %[[VAL_0]][0, 0] : !llvm.struct<(array<2 x i64>, array<3 x i64>)>
// CHECK:           %[[CAST:.*]] = arith.index_cast %[[VAL_1]] : i64 to index
// CHECK:           return %[[CAST]] : index
func.func @sparse_get_md(%arg0: !sparse_tensor.storage_specifier<#CSR>) -> index {
  %0 = sparse_tensor.storage_specifier.get %arg0 lvl_sz at 0
       : !sparse_tensor.storage_specifier<#CSR>
  return %0 : index
}

// CHECK-LABEL:   func.func @sparse_set_md(
// CHECK-SAME:      %[[VAL_0:.*]]: !llvm.struct<(array<2 x i64>, array<3 x i64>)>,
// CHECK-SAME:      %[[VAL_1:.*]]: index) -> !llvm.struct<(array<2 x i64>, array<3 x i64>)> {
// CHECK:           %[[CAST:.*]] = arith.index_cast %[[VAL_1]] : index to i64
// CHECK:           %[[VAL_2:.*]] = llvm.insertvalue %[[CAST]], %[[VAL_0]][0, 0] : !llvm.struct<(array<2 x i64>, array<3 x i64>)>
// CHECK:           return %[[VAL_2]] : !llvm.struct<(array<2 x i64>, array<3 x i64>)>
func.func @sparse_set_md(%arg0: !sparse_tensor.storage_specifier<#CSR>, %arg1: index)
          -> !sparse_tensor.storage_specifier<#CSR> {
  %0 = sparse_tensor.storage_specifier.set %arg0 lvl_sz at 0 with %arg1
       : !sparse_tensor.storage_specifier<#CSR>
  return %0 : !sparse_tensor.storage_specifier<#CSR>
}

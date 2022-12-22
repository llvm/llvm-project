// RUN: mlir-opt %s -sparse-storage-specifier-to-llvm --cse --canonicalize | FileCheck %s

#CSR = #sparse_tensor.encoding<{dimLevelType = ["dense", "compressed"]}>

// CHECK-LABEL:   func.func @sparse_metadata_init() -> !llvm.struct<(array<2 x i64>, array<3 x i64>)> {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.undef : !llvm.struct<(array<2 x i64>, array<3 x i64>)>
// CHECK:           %[[VAL_2:.*]] = llvm.insertvalue %[[VAL_0]], %[[VAL_1]][1, 0] : !llvm.struct<(array<2 x i64>, array<3 x i64>)>
// CHECK:           %[[VAL_3:.*]] = llvm.insertvalue %[[VAL_0]], %[[VAL_2]][1, 1] : !llvm.struct<(array<2 x i64>, array<3 x i64>)>
// CHECK:           %[[VAL_4:.*]] = llvm.insertvalue %[[VAL_0]], %[[VAL_3]][1, 2] : !llvm.struct<(array<2 x i64>, array<3 x i64>)>
// CHECK:           return %[[VAL_4]] : !llvm.struct<(array<2 x i64>, array<3 x i64>)>
// CHECK:         }
func.func @sparse_metadata_init() -> !sparse_tensor.storage_specifier<#CSR> {
  %0 = sparse_tensor.storage_specifier.init : !sparse_tensor.storage_specifier<#CSR>
  return %0 : !sparse_tensor.storage_specifier<#CSR>
}

// CHECK-LABEL:   func.func @sparse_get_md(
// CHECK-SAME:      %[[VAL_0:.*]]: !llvm.struct<(array<2 x i64>, array<3 x i64>)>) -> i64 {
// CHECK:           %[[VAL_1:.*]] = llvm.extractvalue %[[VAL_0]][0, 0] : !llvm.struct<(array<2 x i64>, array<3 x i64>)>
// CHECK:           return %[[VAL_1]] : i64
func.func @sparse_get_md(%arg0: !sparse_tensor.storage_specifier<#CSR>) -> i64 {
  %0 = sparse_tensor.storage_specifier.get %arg0 dim_sz at 0
       : !sparse_tensor.storage_specifier<#CSR> to i64
  return %0 : i64
}

// CHECK-LABEL:   func.func @sparse_set_md(
// CHECK-SAME:      %[[VAL_0:.*]]: !llvm.struct<(array<2 x i64>, array<3 x i64>)>,
// CHECK-SAME:      %[[VAL_1:.*]]: i64) -> !llvm.struct<(array<2 x i64>, array<3 x i64>)> {
// CHECK:           %[[VAL_2:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_0]][0, 0] : !llvm.struct<(array<2 x i64>, array<3 x i64>)>
// CHECK:           return %[[VAL_2]] : !llvm.struct<(array<2 x i64>, array<3 x i64>)>
func.func @sparse_set_md(%arg0: !sparse_tensor.storage_specifier<#CSR>, %arg1: i64)
          -> !sparse_tensor.storage_specifier<#CSR> {
  %0 = sparse_tensor.storage_specifier.set %arg0 dim_sz at 0 with %arg1
       : i64, !sparse_tensor.storage_specifier<#CSR>
  return %0 : !sparse_tensor.storage_specifier<#CSR>
}

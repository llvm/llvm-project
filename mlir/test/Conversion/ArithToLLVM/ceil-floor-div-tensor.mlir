// RUN: mlir-opt -pass-pipeline="builtin.module(func.func(convert-arith-to-llvm))" %s -split-input-file | FileCheck %s
// RUN: mlir-opt --convert-to-llvm="filter-dialects=arith" --split-input-file %s | FileCheck %s

// CHECK-LABEL: @ceildivui_dynamic_tensor
// CHECK-SAME: %[[ARG0:.*]]: tensor<8x4x?xi64>) -> tensor<8x4x?xi64>
func.func @ceildivui_dynamic_tensor(%arg0 : tensor<8x4x?xi64>) -> tensor<8x4x?xi64> {
// CHECK: arith.ceildivui %[[ARG0]], %[[ARG0]] : tensor<8x4x?xi64>
  %0 = arith.ceildivui %arg0, %arg0 : tensor<8x4x?xi64>
  return %0: tensor<8x4x?xi64>
}

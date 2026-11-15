// RUN: mlir-opt --linalg-fuse-elementwise-ops %s | FileCheck %s

// CHECK-LABEL: func.func @expand_after_pad(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]:
// CHECK: %[[COLUMNS:[a-zA-Z0-9_]+]] = arith.addi
// CHECK: %[[EXPAND:[a-zA-Z0-9_]+]] = tensor.expand_shape %[[INPUT]] {{.*}}output_shape [2, %[[COLUMNS]]]
// CHECK: tensor.pad %[[EXPAND]]
func.func @expand_after_pad(%input: tensor<?xf32>, %n: index) -> tensor<2x?xf32> {
  %zero = arith.constant 0.0 : f32
  %pad = tensor.pad %input nofold low[0] high[0] {
  ^bb0(%i: index):
    tensor.yield %zero : f32
  } : tensor<?xf32> to tensor<?xf32>
  %columns = arith.addi %n, %n : index
  %expand = tensor.expand_shape %pad [[0, 1]] output_shape [2, %columns]
      : tensor<?xf32> into tensor<2x?xf32>
  return %expand : tensor<2x?xf32>
}

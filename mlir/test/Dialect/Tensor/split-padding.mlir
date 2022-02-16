// RUN: mlir-opt -split-input-file -test-tensor-transform-patterns=test-split-padding-patterns %s | FileCheck %s

// CHECK-LABEL: func @pad_all_zero_sizes
func @pad_all_zero_sizes(%input: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %f0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %0 = tensor.pad %input low[0, %c0, 0] high[%c0, 0, 0] {
  ^bb0(%dim0: index, %dim1: index, %dim2: index):
    tensor.yield %f0 : f32
  } : tensor<?x?x?xf32> to tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}

// CHECK-NOT: scf.if
//     CHECK: tensor.pad

// -----

// CHECK-LABEL: func @pad_non_zero_sizes
//  CHECK-SAME: (%[[INPUT:.+]]: tensor<?x?x8xf32>, %[[LOW0:.+]]: index, %[[HIGH1:.+]]: index)
func @pad_non_zero_sizes(%input: tensor<?x?x8xf32>, %low0: index, %high1: index) -> tensor<?x?x8xf32> {
  %f0 = arith.constant 0.0 : f32
  %0 = tensor.pad %input low[%low0, 0, 0] high[0, %high1, 0] {
  ^bb0(%dim0: index, %dim1: index, %dim2: index):
    tensor.yield %f0 : f32
  } : tensor<?x?x8xf32> to tensor<?x?x8xf32>
  return %0 : tensor<?x?x8xf32>
}

// CHECK: %[[F0:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[C0:.+]] = arith.constant 0 : index
// CHECK: %[[EQ0:.+]] = arith.cmpi eq, %[[LOW0]], %[[C0]] : index
// CHECK: %[[EQ1:.+]] = arith.cmpi eq, %[[HIGH1]], %[[C0]] : index
// CHECK: %[[AND:.+]] = arith.andi %[[EQ0]], %[[EQ1]] : i1
// CHECK: %[[IF:.+]] = scf.if %[[AND]] -> (tensor<?x?x8xf32>) {
// CHECK:   scf.yield %[[INPUT]] : tensor<?x?x8xf32>
// CHECK: } else {
// CHECK:   %[[PAD:.+]] = tensor.pad %[[INPUT]] low[%[[LOW0]], 0, 0] high[0, %[[HIGH1]], 0]  {
// CHECK:   ^bb0(%{{.+}}: index, %{{.+}}: index, %{{.+}}: index):
// CHECK:     tensor.yield %[[F0]] : f32
// CHECK:   } : tensor<?x?x8xf32> to tensor<?x?x8xf32>
// CHECK:   scf.yield %[[PAD]] : tensor<?x?x8xf32>
// CHECK: }
// CHECK: return %[[IF]] : tensor<?x?x8xf32>

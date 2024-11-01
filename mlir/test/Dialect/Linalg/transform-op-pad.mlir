// RUN: mlir-opt --test-transform-dialect-interpreter -split-input-file -verify-diagnostics %s | FileCheck %s

#map = affine_map<()[s0] -> (-s0 + 12, 7)>

// CHECK-LABEL: @static_sizes_output_divisible
func.func @static_sizes_output_divisible(%arg0: tensor<24x12xf32>,
                                         %arg1: tensor<12x25xf32>,
                                         %arg2: tensor<24x25xf32>,
                                         %iv0 : index, %iv1 : index, %iv2 : index) -> tensor<24x25xf32> {
  %0 = affine.min #map()[%iv2]

  //      CHECK: %[[T0:.*]] = tensor.extract_slice %
  //      CHECK: %[[T1:.*]] = tensor.extract_slice %
  //      CHECK: %[[T2:.*]] = tensor.extract_slice %
  %1 = tensor.extract_slice %arg0[%iv0, %iv2] [4, %0] [1, 1] : tensor<24x12xf32> to tensor<4x?xf32>
  %2 = tensor.extract_slice %arg1[%iv2, %iv1] [%0, 5] [1, 1] : tensor<12x25xf32> to tensor<?x5xf32>
  %3 = tensor.extract_slice %arg2[%iv0, %iv1] [4, 5] [1, 1] : tensor<24x25xf32> to tensor<4x5xf32>

  //  CHECK-DAG: %[[CST:.*]] = arith.constant 0.
  //  CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index

  //      CHECK: %[[T3:.*]] = tensor.pad %[[T0]] nofold
  //      CHECK: tensor.yield %[[CST]]
  //      CHECK: %[[T4:.*]] = tensor.pad %[[T1]] nofold

  //      CHECK: %[[T5:.*]] = linalg.matmul
  // CHECK-SAME:              ins(%[[T3]], %[[T4]] : tensor<4x7xf32>, tensor<7x5xf32>)
  // CHECK-SAME:              outs(%[[T2]] : tensor<4x5xf32>)
  %4 = linalg.matmul ins(%1, %2 : tensor<4x?xf32>, tensor<?x5xf32>) outs(%3 : tensor<4x5xf32>) -> tensor<4x5xf32>
  %5 = tensor.insert_slice %4 into %arg2[%iv0, %iv1] [4, 5] [1, 1] : tensor<4x5xf32> into tensor<24x25xf32>
  func.return %5 : tensor<24x25xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
  %1 = transform.structured.pad %0 {padding_values=[0.0 : f32, 0.0 : f32, 0.0 : f32], padding_dimensions=[0, 1, 2], pack_paddings=[1, 1, 0]}
}

// -----

func.func @pad(%arg0: tensor<24x12xf32>,
               %arg1: tensor<12x25xf32>,
               %arg2: tensor<24x25xf32>) -> tensor<24x25xf32> {
  // expected-note @below {{when applied to this op}}
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<24x12xf32>, tensor<12x25xf32>) outs(%arg2 : tensor<24x25xf32>) -> tensor<24x25xf32>
  func.return %0 : tensor<24x25xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
  // expected-error @below {{op expects a padding value of type 'f32', got 0 : i32}}
  %1 = transform.structured.pad %0 {padding_values=[0: i32, 0.0 : f32, 0.0 : f32], padding_dimensions=[0, 1, 2], pack_paddings=[1, 1, 0]}
}

// -----

func.func @pad(%arg0: tensor<24x12xf32>,
               %arg1: tensor<12x25xf32>,
               %arg2: tensor<24x25xf32>) -> tensor<24x25xf32> {
  // expected-note @below {{when applied to this op}}
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<24x12xf32>, tensor<12x25xf32>) outs(%arg2 : tensor<24x25xf32>) -> tensor<24x25xf32>
  func.return %0 : tensor<24x25xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
  // expected-error @below {{expects a padding that parses to 'f32', got "foo"}}
  %1 = transform.structured.pad %0 {padding_values=["foo", 0.0 : f32, 0.0 : f32], padding_dimensions=[0, 1, 2], pack_paddings=[1, 1, 0]}
}

// -----

func.func @pad(%arg0: tensor<24x12xf32>,
               %arg1: tensor<12x25xf32>,
               %arg2: tensor<24x25xf32>) -> tensor<24x25xf32> {
  // This is attached to an error that is silenceable and is not reported by this transform
  //   {{when applied to this op}}
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<24x12xf32>, tensor<12x25xf32>) outs(%arg2 : tensor<24x25xf32>) -> tensor<24x25xf32>
  func.return %0 : tensor<24x25xf32>
}

transform.sequence failures(suppress) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
  // This error is silenceable and is not reported by this transform
  //   {{transform.structured.pad failed to apply}}
  %1 = transform.structured.pad %0 {padding_values=[0.0 : f32, 0.0 : f32, 0.0 : f32], padding_dimensions=[0, 1, 2], pack_paddings=[1, 1, 0]}
}

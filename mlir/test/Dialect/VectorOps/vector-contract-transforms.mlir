// RUN: mlir-opt %s -test-vector-contraction-conversion | FileCheck %s

#dotp_accesses = [
  affine_map<(i) -> (i)>,
  affine_map<(i) -> (i)>,
  affine_map<(i) -> ()>
]
#dotp_trait = {
  indexing_maps = #dotp_accesses,
  iterator_types = ["reduction"]
}

// CHECK-LABEL: func @extract_contract1
// CHECK-SAME: %[[A:.*0]]: vector<4xf32>,
// CHECK-SAME: %[[B:.*1]]: vector<4xf32>,
// CHECK-SAME: %[[C:.*2]]: f32
// CHECK:      %[[Z:.*]] = constant dense<0.000000e+00>
// CHECK:      %[[F:.*]] = vector.fma %[[A]], %[[B]], %[[Z]] : vector<4xf32>
// CHECK:      %[[R:.*]] = vector.reductionv2 "add", %[[F]], %[[C]]
// CHECK:      return %[[R]] : f32

func @extract_contract1(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %arg2: f32) -> f32 {
  %0 = vector.contract #dotp_trait %arg0, %arg1, %arg2
    : vector<4xf32>, vector<4xf32> into f32
  return %0 : f32
}

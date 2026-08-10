// RUN: mlir-opt %s --transform-interpreter | FileCheck %s

// CHECK-LABEL: @vector_deinterleave_2d
// CHECK-SAME: %[[SRC:.*]]: vector<2x8xi32>) -> (vector<2x4xi32>, vector<2x4xi32>)
func.func @vector_deinterleave_2d(%a: vector<2x8xi32>) -> (vector<2x4xi32>, vector<2x4xi32>) {
  // CHECK: %[[CST:.*]] = arith.constant dense<0>
  // CHECK: %[[SRC_0:.*]] = vector.extract %[[SRC]][0]
  // CHECK: %[[UNZIP_0:.*]], %[[UNZIP_1:.*]] = vector.deinterleave %[[SRC_0]]
  // CHECK: %[[RES_0:.*]] = vector.insert %[[UNZIP_0]], %[[CST]] [0]
  // CHECK: %[[RES_1:.*]] = vector.insert %[[UNZIP_1]], %[[CST]] [0]
  // CHECK: %[[SRC_1:.*]] = vector.extract %[[SRC]][1]
  // CHECK: %[[UNZIP_2:.*]], %[[UNZIP_3:.*]] = vector.deinterleave %[[SRC_1]]
  // CHECK: %[[RES_2:.*]] = vector.insert %[[UNZIP_2]], %[[RES_0]] [1]
  // CHECK: %[[RES_3:.*]] = vector.insert %[[UNZIP_3]], %[[RES_1]] [1]
  // CHECK-NEXT: return %[[RES_2]], %[[RES_3]] : vector<2x4xi32>, vector<2x4xi32>
  %0, %1 = vector.deinterleave %a : vector<2x8xi32> -> vector<2x4xi32>
  return %0, %1 : vector<2x4xi32>, vector<2x4xi32>
}

// CHECK-LABEL: @vector_deinterleave_2d_scalable
// CHECK-SAME: %[[SRC:.*]]: vector<2x[8]xi32>) -> (vector<2x[4]xi32>, vector<2x[4]xi32>)
func.func @vector_deinterleave_2d_scalable(%a: vector<2x[8]xi32>) -> (vector<2x[4]xi32>, vector<2x[4]xi32>) {
  // CHECK: %[[CST:.*]] = arith.constant dense<0>
  // CHECK: %[[SRC_0:.*]] = vector.extract %[[SRC]][0]
  // CHECK: %[[UNZIP_0:.*]], %[[UNZIP_1:.*]] = vector.deinterleave %[[SRC_0]]
  // CHECK: %[[RES_0:.*]] = vector.insert %[[UNZIP_0]], %[[CST]] [0]
  // CHECK: %[[RES_1:.*]] = vector.insert %[[UNZIP_1]], %[[CST]] [0]
  // CHECK: %[[SRC_1:.*]] = vector.extract %[[SRC]][1]
  // CHECK: %[[UNZIP_2:.*]], %[[UNZIP_3:.*]] = vector.deinterleave %[[SRC_1]]
  // CHECK: %[[RES_2:.*]] = vector.insert %[[UNZIP_2]], %[[RES_0]] [1]
  // CHECK: %[[RES_3:.*]] = vector.insert %[[UNZIP_3]], %[[RES_1]] [1]
  // CHECK-NEXT: return %[[RES_2]], %[[RES_3]] : vector<2x[4]xi32>, vector<2x[4]xi32>
  %0, %1 = vector.deinterleave %a : vector<2x[8]xi32> -> vector<2x[4]xi32>
  return %0, %1 : vector<2x[4]xi32>, vector<2x[4]xi32>
}

// CHECK-LABEL: @vector_deinterleave_4d
// CHECK-SAME: %[[SRC:.*]]: vector<1x2x3x8xi64>) -> (vector<1x2x3x4xi64>, vector<1x2x3x4xi64>)
func.func @vector_deinterleave_4d(%a: vector<1x2x3x8xi64>) -> (vector<1x2x3x4xi64>, vector<1x2x3x4xi64>) {
    // CHECK: %[[SRC_0:.*]] = vector.extract %[[SRC]][0, 0, 0] : vector<8xi64> from vector<1x2x3x8xi64>
    // CHECK: %[[UNZIP_0:.*]], %[[UNZIP_1:.*]] = vector.deinterleave %[[SRC_0]] : vector<8xi64> -> vector<4xi64>
    // CHECK: %[[RES_0:.*]] = vector.insert %[[UNZIP_0]], %{{.*}} [0, 0, 0] : vector<4xi64> into vector<1x2x3x4xi64>
    // CHECK: %[[RES_1:.*]] = vector.insert %[[UNZIP_1]], %{{.*}} [0, 0, 0] : vector<4xi64> into vector<1x2x3x4xi64>
    // CHECK-COUNT-5: vector.deinterleave %{{.*}} : vector<8xi64> -> vector<4xi64>
    %0, %1 = vector.deinterleave %a : vector<1x2x3x8xi64> -> vector<1x2x3x4xi64>
    return %0, %1 : vector<1x2x3x4xi64>, vector<1x2x3x4xi64>
}

// CHECK-LABEL: @vector_deinterleave_nd_with_scalable_dim
func.func @vector_deinterleave_nd_with_scalable_dim(
  %a: vector<1x3x[2]x2x3x8xf16>) -> (vector<1x3x[2]x2x3x4xf16>, vector<1x3x[2]x2x3x4xf16>) {
  // The scalable dim blocks unrolling so only the first two dims are unrolled.
  // CHECK-COUNT-3: vector.deinterleave %{{.*}} : vector<[2]x2x3x8xf16>
  %0, %1 = vector.deinterleave %a: vector<1x3x[2]x2x3x8xf16> -> vector<1x3x[2]x2x3x4xf16>
  return %0, %1 : vector<1x3x[2]x2x3x4xf16>, vector<1x3x[2]x2x3x4xf16>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %module_op
      : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %f {
      transform.apply_patterns.vector.lower_interleave
    } : !transform.any_op
    transform.yield
  }
}

// RUN: mlir-opt %s --transform-interpreter | FileCheck %s

// CHECK-LABEL: @vector_interleave_2d
//  CHECK-SAME:     %[[LHS:.*]]: vector<2x3xi8>, %[[RHS:.*]]: vector<2x3xi8>)
func.func @vector_interleave_2d(%a: vector<2x3xi8>, %b: vector<2x3xi8>) -> vector<2x6xi8> {
  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<0>
  // CHECK-DAG: %[[LHS_0:.*]] = vector.extract %[[LHS]][0]
  // CHECK-DAG: %[[RHS_0:.*]] = vector.extract %[[RHS]][0]
  // CHECK-DAG: %[[LHS_1:.*]] = vector.extract %[[LHS]][1]
  // CHECK-DAG: %[[RHS_1:.*]] = vector.extract %[[RHS]][1]
  // CHECK-DAG: %[[ZIP_0:.*]] = vector.interleave %[[LHS_0]], %[[RHS_0]]
  // CHECK-DAG: %[[ZIP_1:.*]] = vector.interleave %[[LHS_1]], %[[RHS_1]]
  // CHECK-DAG: %[[RES_0:.*]] = vector.insert %[[ZIP_0]], %[[CST]] [0]
  // CHECK-DAG: %[[RES_1:.*]] = vector.insert %[[ZIP_1]], %[[RES_0]] [1]
  // CHECK-NEXT: return %[[RES_1]] : vector<2x6xi8>
  %0 = vector.interleave %a, %b : vector<2x3xi8> -> vector<2x6xi8>
  return %0 : vector<2x6xi8>
}

// CHECK-LABEL: @vector_interleave_2d_scalable
//  CHECK-SAME:     %[[LHS:.*]]: vector<2x[8]xi16>, %[[RHS:.*]]: vector<2x[8]xi16>)
func.func @vector_interleave_2d_scalable(%a: vector<2x[8]xi16>, %b: vector<2x[8]xi16>) -> vector<2x[16]xi16> {
  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<0>
  // CHECK-DAG: %[[LHS_0:.*]] = vector.extract %[[LHS]][0]
  // CHECK-DAG: %[[RHS_0:.*]] = vector.extract %[[RHS]][0]
  // CHECK-DAG: %[[LHS_1:.*]] = vector.extract %[[LHS]][1]
  // CHECK-DAG: %[[RHS_1:.*]] = vector.extract %[[RHS]][1]
  // CHECK-DAG: %[[ZIP_0:.*]] = vector.interleave %[[LHS_0]], %[[RHS_0]]
  // CHECK-DAG: %[[ZIP_1:.*]] = vector.interleave %[[LHS_1]], %[[RHS_1]]
  // CHECK-DAG: %[[RES_0:.*]] = vector.insert %[[ZIP_0]], %[[CST]] [0]
  // CHECK-DAG: %[[RES_1:.*]] = vector.insert %[[ZIP_1]], %[[RES_0]] [1]
  // CHECK-NEXT: return %[[RES_1]] : vector<2x[16]xi16>
  %0 = vector.interleave %a, %b : vector<2x[8]xi16> -> vector<2x[16]xi16>
  return %0 : vector<2x[16]xi16>
}

// CHECK-LABEL: @vector_interleave_4d
//  CHECK-SAME:     %[[LHS:.*]]: vector<1x2x3x4xi64>, %[[RHS:.*]]: vector<1x2x3x4xi64>)
func.func @vector_interleave_4d(%a: vector<1x2x3x4xi64>, %b: vector<1x2x3x4xi64>) -> vector<1x2x3x8xi64>
{
  // CHECK: %[[LHS_0:.*]] = vector.extract %[[LHS]][0, 0, 0] : vector<4xi64> from vector<1x2x3x4xi64>
  // CHECK: %[[RHS_0:.*]] = vector.extract %[[RHS]][0, 0, 0] : vector<4xi64> from vector<1x2x3x4xi64>
  // CHECK: %[[ZIP_0:.*]] = vector.interleave %[[LHS_0]], %[[RHS_0]] : vector<4xi64>
  // CHECK: %[[RES_0:.*]] = vector.insert %[[ZIP_0]], %{{.*}} [0, 0, 0] : vector<8xi64> into vector<1x2x3x8xi64>
  // CHECK-COUNT-5: vector.interleave %{{.*}}, %{{.*}} : vector<4xi64> -> vector<8xi64>
  %0 = vector.interleave %a, %b : vector<1x2x3x4xi64> -> vector<1x2x3x8xi64>
  return %0 : vector<1x2x3x8xi64>
}

// CHECK-LABEL: @vector_interleave_nd_with_scalable_dim
func.func @vector_interleave_nd_with_scalable_dim(
  %a: vector<1x3x[2]x2x3x4xf16>, %b: vector<1x3x[2]x2x3x4xf16>) -> vector<1x3x[2]x2x3x8xf16> {
  // The scalable dim blocks unrolling so only the first two dims are unrolled.
  // CHECK-COUNT-3: vector.interleave %{{.*}}, %{{.*}} : vector<[2]x2x3x4xf16>
  %0 = vector.interleave %a, %b : vector<1x3x[2]x2x3x4xf16> -> vector<1x3x[2]x2x3x8xf16>
  return %0 : vector<1x3x[2]x2x3x8xf16>
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

// RUN: mlir-opt %s --transform-interpreter | FileCheck %s

//===----------------------------------------------------------------------===//
// Test UnrollVectorMultiReduction for Inner Reduction
//===----------------------------------------------------------------------===//
//
// The general case handles multiple reduction dimensions.
// For vector<2x3x5xf32> with reduction on dims [1, 2]:
// UnrollMultiReductionInnerReduction unrolls along dim 0 (size 2), creating
// two vector<3x5xf32> multi_reductions with dims [0, 1], then insert results.

// CHECK-LABEL: func @unroll_vector_multi_reduction_inner_general(
// CHECK-SAME: %[[SOURCE:.+]]: vector<2x3x5xf32>,
// CHECK-SAME: %[[ACC:.+]]: vector<2xf32>
func.func @unroll_vector_multi_reduction_inner_general(%source: vector<2x3x5xf32>, %acc: vector<2xf32>) -> (vector<2xf32>) {
  // CHECK-DAG: %[[SRC_0:.+]] = vector.extract %[[SOURCE]][0] : vector<3x5xf32> from vector<2x3x5xf32>
  // CHECK-DAG: %[[SRC_1:.+]] = vector.extract %[[SOURCE]][1] : vector<3x5xf32> from vector<2x3x5xf32>
  // CHECK-DAG: %[[ACC_0:.+]] = vector.extract %[[ACC]][0] : f32 from vector<2xf32>
  // CHECK-DAG: %[[ACC_1:.+]] = vector.extract %[[ACC]][1] : f32 from vector<2xf32>
  // CHECK: %[[R0:.+]] = vector.multi_reduction <add>, %[[SRC_0]], %[[ACC_0]] [0, 1] : vector<3x5xf32> to f32
  // CHECK: %[[R1:.+]] = vector.multi_reduction <add>, %[[SRC_1]], %[[ACC_1]] [0, 1] : vector<3x5xf32> to f32
  // CHECK: %[[INSERT_0:.+]] = vector.insert %[[R0]], %{{.+}} [0] : f32 into vector<2xf32>
  // CHECK: %[[INSERT_1:.+]] = vector.insert %[[R1]], %[[INSERT_0]] [1] : f32 into vector<2xf32>
  %1 = vector.multi_reduction <add>, %source, %acc [1, 2] : vector<2x3x5xf32> to vector<2xf32>

  // CHECK: return %[[INSERT_1]]
  return %1 : vector<2xf32>
}

// CHECK-LABEL: func @unroll_vector_multi_reduction_inner_general_masked(
// CHECK-SAME: %[[SOURCE:.+]]: vector<2x3x5xf32>,
// CHECK-SAME: %[[MASK:.+]]: vector<2x3x5xi1>,
// CHECK-SAME: %[[ACC:.+]]: vector<2xf32>
func.func @unroll_vector_multi_reduction_inner_general_masked(%source: vector<2x3x5xf32>, %mask: vector<2x3x5xi1>, %acc: vector<2xf32>) -> (vector<2xf32>) {
  // CHECK-DAG: %[[SRC_0:.+]] = vector.extract %[[SOURCE]][0] : vector<3x5xf32> from vector<2x3x5xf32>
  // CHECK-DAG: %[[SRC_1:.+]] = vector.extract %[[SOURCE]][1] : vector<3x5xf32> from vector<2x3x5xf32>
  // CHECK-DAG: %[[ACC_0:.+]] = vector.extract %[[ACC]][0] : f32 from vector<2xf32>
  // CHECK-DAG: %[[ACC_1:.+]] = vector.extract %[[ACC]][1] : f32 from vector<2xf32>
  // CHECK-DAG: %[[MASK_0:.+]] = vector.extract %[[MASK]][0] : vector<3x5xi1> from vector<2x3x5xi1>
  // CHECK-DAG: %[[MASK_1:.+]] = vector.extract %[[MASK]][1] : vector<3x5xi1> from vector<2x3x5xi1>
  // CHECK: %[[R0:.+]] = vector.mask %[[MASK_0]] {{.*}} vector.multi_reduction <add>, %[[SRC_0]], %[[ACC_0]] [0, 1] : vector<3x5xf32> to f32
  // CHECK: %[[R1:.+]] = vector.mask %[[MASK_1]] {{.*}} vector.multi_reduction <add>, %[[SRC_1]], %[[ACC_1]] [0, 1] : vector<3x5xf32> to f32
  // CHECK: %[[INSERT_0:.+]] = vector.insert %[[R0]], %{{.+}} [0] : f32 into vector<2xf32>
  // CHECK: %[[INSERT_1:.+]] = vector.insert %[[R1]], %[[INSERT_0]] [1] : f32 into vector<2xf32>

  %0 = vector.mask %mask {
    %1 = vector.multi_reduction <add>, %source, %acc [1, 2] : vector<2x3x5xf32> to vector<2xf32>
  } : vector<2x3x5xi1> -> vector<2xf32>

  // CHECK: return %[[INSERT_1]]
  return %0 : vector<2xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.multi_reduction_unrolling lowering_strategy = innerreduction
    } : !transform.op<"func.func">
    transform.yield
  }
}

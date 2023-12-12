// RUN: mlir-opt %s --transform-interpreter --split-input-file | FileCheck %s

// CHECK-LABEL: func @outerproduct_noacc
// CHECK-SAME: %[[A:.*0]]: vector<2xf32>,
// CHECK-SAME: %[[B:.*1]]: vector<3xf32>
// CHECK:      %[[C0:.*]] = arith.constant dense<0.000000e+00> : vector<2x3xf32>
// CHECK:      %[[T0:.*]] = vector.extract %[[A]][0] : f32 from vector<2xf32>
// CHECK:      %[[T1:.*]] = vector.splat %[[T0]] : vector<3xf32>
// CHECK:      %[[T2:.*]] = arith.mulf %[[T1]], %[[B]] : vector<3xf32>
// CHECK:      %[[T3:.*]] = vector.insert %[[T2]], %[[C0]] [0] : vector<3xf32> into vector<2x3xf32>
// CHECK:      %[[T4:.*]] = vector.extract %[[A]][1] : f32 from vector<2xf32>
// CHECK:      %[[T5:.*]] = vector.splat %[[T4]] : vector<3xf32>
// CHECK:      %[[T6:.*]] = arith.mulf %[[T5]], %[[B]] : vector<3xf32>
// CHECK:      %[[T7:.*]] = vector.insert %[[T6]], %[[T3]] [1] : vector<3xf32> into vector<2x3xf32>
// CHECK:      return %[[T7]] : vector<2x3xf32>

func.func @outerproduct_noacc(%arg0: vector<2xf32>,
                              %arg1: vector<3xf32>) -> vector<2x3xf32> {
  %0 = vector.outerproduct %arg0, %arg1 : vector<2xf32>, vector<3xf32>
  return %0: vector<2x3xf32>
}

// CHECK-LABEL: func @outerproduct_acc
// CHECK-SAME: %[[A:.*0]]: vector<2xf32>,
// CHECK-SAME: %[[B:.*1]]: vector<3xf32>,
// CHECK-SAME: %[[C:.*2]]: vector<2x3xf32>
// CHECK:      %[[C0:.*]] = arith.constant dense<0.000000e+00> : vector<2x3xf32>
// CHECK:      %[[T0:.*]] = vector.extract %[[A]][0] : f32 from vector<2xf32>
// CHECK:      %[[T1:.*]] = vector.splat %[[T0]] : vector<3xf32>
// CHECK:      %[[T2:.*]] = vector.extract %[[C]][0] : vector<3xf32> from vector<2x3xf32>
// CHECK:      %[[T3:.*]] = vector.fma %[[T1]], %[[B]], %[[T2]] : vector<3xf32>
// CHECK:      %[[T4:.*]] = vector.insert %[[T3]], %[[C0]] [0] : vector<3xf32> into vector<2x3xf32>
// CHECK:      %[[T5:.*]] = vector.extract %[[A]][1] : f32 from vector<2xf32>
// CHECK:      %[[T6:.*]] = vector.splat %[[T5]] : vector<3xf32>
// CHECK:      %[[T7:.*]] = vector.extract %[[C]][1] : vector<3xf32> from vector<2x3xf32>
// CHECK:      %[[T8:.*]] = vector.fma %[[T6]], %[[B]], %[[T7]] : vector<3xf32>
// CHECK:      %[[T9:.*]] = vector.insert %[[T8]], %[[T4]] [1] : vector<3xf32> into vector<2x3xf32>
// CHECK:      return %[[T9]] : vector<2x3xf32>

func.func @outerproduct_acc(%arg0: vector<2xf32>,
                            %arg1: vector<3xf32>,
                            %arg2: vector<2x3xf32>) -> vector<2x3xf32> {
  %0 = vector.outerproduct %arg0, %arg1, %arg2 : vector<2xf32>, vector<3xf32>
  return %0: vector<2x3xf32>
}

// CHECK-LABEL: func @outerproduct_noacc_int
// CHECK-SAME: %[[A:.*0]]: vector<2xi32>,
// CHECK-SAME: %[[B:.*1]]: vector<3xi32>
// CHECK:      %[[C0:.*]] = arith.constant dense<0> : vector<2x3xi32>
// CHECK:      %[[T0:.*]] = vector.extract %[[A]][0] : i32 from vector<2xi32>
// CHECK:      %[[T1:.*]] = vector.splat %[[T0]] : vector<3xi32>
// CHECK:      %[[T2:.*]] = arith.muli %[[T1]], %[[B]] : vector<3xi32>
// CHECK:      %[[T3:.*]] = vector.insert %[[T2]], %[[C0]] [0] : vector<3xi32> into vector<2x3xi32>
// CHECK:      %[[T4:.*]] = vector.extract %[[A]][1] : i32 from vector<2xi32>
// CHECK:      %[[T5:.*]] = vector.splat %[[T4]] : vector<3xi32>
// CHECK:      %[[T6:.*]] = arith.muli %[[T5]], %[[B]] : vector<3xi32>
// CHECK:      %[[T7:.*]] = vector.insert %[[T6]], %[[T3]] [1] : vector<3xi32> into vector<2x3xi32>
// CHECK:      return %[[T7]] : vector<2x3xi32>
func.func @outerproduct_noacc_int(%arg0: vector<2xi32>,
                                  %arg1: vector<3xi32>) -> vector<2x3xi32> {
  %0 = vector.outerproduct %arg0, %arg1 : vector<2xi32>, vector<3xi32>
  return %0: vector<2x3xi32>
}

// CHECK-LABEL: func @outerproduct_acc_int
// CHECK-SAME: %[[A:.*0]]: vector<2xi32>,
// CHECK-SAME: %[[B:.*1]]: vector<3xi32>,
// CHECK-SAME: %[[C:.*2]]: vector<2x3xi32>
// CHECK:      %[[C0:.*]] = arith.constant dense<0> : vector<2x3xi32>
// CHECK:      %[[T0:.*]] = vector.extract %[[A]][0] : i32 from vector<2xi32>
// CHECK:      %[[T1:.*]] = vector.splat %[[T0]] : vector<3xi32>
// CHECK:      %[[T2:.*]] = vector.extract %[[C]][0] : vector<3xi32> from vector<2x3xi32>
// CHECK:      %[[T3:.*]] = arith.muli %[[T1]], %[[B]] : vector<3xi32>
// CHECK:      %[[T4:.*]] = arith.addi %[[T3]], %[[T2]] : vector<3xi32>
// CHECK:      %[[T5:.*]] = vector.insert %[[T4]], %[[C0]] [0] : vector<3xi32> into vector<2x3xi32>
// CHECK:      %[[T6:.*]] = vector.extract %[[A]][1] : i32 from vector<2xi32>
// CHECK:      %[[T7:.*]] = vector.splat %[[T6]] : vector<3xi32>
// CHECK:      %[[T8:.*]] = vector.extract %[[C]][1] : vector<3xi32> from vector<2x3xi32>
// CHECK:      %[[T9:.*]] = arith.muli %[[T7]], %[[B]] : vector<3xi32>
// CHECK:      %[[T10:.*]] = arith.addi %[[T9]], %[[T8]] : vector<3xi32>
// CHECK:      %[[T11:.*]] = vector.insert %[[T10]], %[[T5]] [1] : vector<3xi32> into vector<2x3xi32>
// CHECK:      return %[[T11]] : vector<2x3xi32>
func.func @outerproduct_acc_int(%arg0: vector<2xi32>,
                                %arg1: vector<3xi32>,
                                %arg2: vector<2x3xi32>) -> vector<2x3xi32> {
  %0 = vector.outerproduct %arg0, %arg1, %arg2 : vector<2xi32>, vector<3xi32>
  return %0: vector<2x3xi32>
}

// CHECK-LABEL: func @axpy_fp(
// CHECK-SAME: %[[A:.*0]]: vector<16xf32>,
// CHECK-SAME: %[[B:.*1]]: f32)
// CHECK: %[[T0:.*]] = vector.splat %[[B]] : vector<16xf32>
// CHECK: %[[T1:.*]] = arith.mulf %[[A]], %[[T0]] : vector<16xf32>
// CHECK: return %[[T1]] : vector<16xf32>
func.func @axpy_fp(%arg0: vector<16xf32>, %arg1: f32) -> vector<16xf32> {
  %0 = vector.outerproduct %arg0, %arg1: vector<16xf32>, f32
  return %0: vector<16xf32>
}

// CHECK-LABEL: func @axpy_fp_add(
// CHECK-SAME: %[[A:.*0]]: vector<16xf32>,
// CHECK-SAME: %[[B:.*1]]: f32,
// CHECK-SAME: %[[C:.*2]]: vector<16xf32>)
// CHECK: %[[T0:.*]] = vector.splat %[[B]] : vector<16xf32>
// CHECK: %[[T1:.*]] = vector.fma %[[A]], %[[T0]], %[[C]] : vector<16xf32>
// CHECK: return %[[T1]] : vector<16xf32>
func.func @axpy_fp_add(%arg0: vector<16xf32>, %arg1: f32, %arg2 : vector<16xf32>) -> vector<16xf32> {
  %0 = vector.outerproduct %arg0, %arg1, %arg2: vector<16xf32>, f32
  return %0: vector<16xf32>
}

// CHECK-LABEL: func @axpy_int(
// CHECK-SAME: %[[A:.*0]]: vector<16xi32>,
// CHECK-SAME: %[[B:.*1]]: i32)
// CHECK: %[[T0:.*]] = vector.splat %[[B]] : vector<16xi32>
// CHECK: %[[T1:.*]] = arith.muli %[[A]], %[[T0]] : vector<16xi32>
// CHECK: return %[[T1]] : vector<16xi32>
func.func @axpy_int(%arg0: vector<16xi32>, %arg1: i32) -> vector<16xi32> {
  %0 = vector.outerproduct %arg0, %arg1: vector<16xi32>, i32
  return %0: vector<16xi32>
}

// CHECK-LABEL: func @axpy_int_add(
// CHECK-SAME: %[[A:.*0]]: vector<16xi32>,
// CHECK-SAME: %[[B:.*1]]: i32,
// CHECK-SAME: %[[C:.*2]]: vector<16xi32>)
// CHECK: %[[T0:.*]] = vector.splat %[[B]] : vector<16xi32>
// CHECK: %[[T1:.*]] = arith.muli %[[A]], %[[T0]] : vector<16xi32>
// CHECK: %[[T2:.*]] = arith.addi %[[T1]], %[[C]] : vector<16xi32>
// CHECK: return %[[T2]] : vector<16xi32>
func.func @axpy_int_add(%arg0: vector<16xi32>, %arg1: i32, %arg2: vector<16xi32>) -> vector<16xi32> {
  %0 = vector.outerproduct %arg0, %arg1, %arg2: vector<16xi32>, i32
  return %0: vector<16xi32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %module_op
      : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %f {
      transform.apply_patterns.vector.lower_outerproduct
    } : !transform.any_op

    transform.apply_patterns to %f {
      transform.apply_patterns.vector.lower_broadcast
    } : !transform.any_op
    transform.yield
  }
}

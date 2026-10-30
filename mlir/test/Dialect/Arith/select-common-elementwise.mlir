// RUN: mlir-opt %s --canonicalize="test-convergence" | FileCheck %s

// Test select(M, F(a0, a1, ..., a_n), F(b0, b1, ..., b_n)) -> F(select(M, a0, b0), select(M, a1, b1), ...).

// CHECK-LABEL: func.func @basic(
// CHECK-NEXT: %[[A:.*]] = arith.select %arg0, %arg1, %arg3 : tensor<2x3xi1>, tensor<2x3xi32>
// CHECK-NEXT: %[[B:.*]] = arith.select %arg0, %arg2, %arg4 : tensor<2x3xi1>, tensor<2x3xi32>
// CHECK-NEXT: %[[M:.*]] = arith.subi %[[A]], %[[B]] : tensor<2x3xi32>
// CHECK-NEXT: %[[R:.*]] = arith.extsi %[[M]] : tensor<2x3xi32> to tensor<2x3xi64>
// CHECK-NEXT: return %[[R]] : tensor<2x3xi64>
func.func @basic(%mask: tensor<2x3xi1>, %a0: tensor<2x3xi32>, %b0: tensor<2x3xi32>, %a1: tensor<2x3xi32>, %b1: tensor<2x3xi32>) -> tensor<2x3xi64> {
  %x = arith.subi %a0, %b0 : tensor<2x3xi32>
  %y = arith.subi %a1, %b1 : tensor<2x3xi32>
  %ex = arith.extsi %x : tensor<2x3xi32> to tensor<2x3xi64>
  %ey = arith.extsi %y : tensor<2x3xi32> to tensor<2x3xi64>
  %r = arith.select %mask, %ex, %ey : tensor<2x3xi1>, tensor<2x3xi64>
  return %r : tensor<2x3xi64>
}

// If the operands of the select op are not the same, do not transform
// CHECK-LABEL: func.func @diff_op(
// CHECK-NEXT: %[[X:.*]] = arith.muli %arg1, %arg3 overflow<nsw> : i32
// CHECK-NEXT: %[[Y:.*]] = arith.muli %arg2, %arg3 overflow<nuw> : i32
// CHECK-NEXT: %[[R:.*]] = arith.select %arg0, %[[X]], %[[Y]] : i32
// CHECK-NEXT: return %[[R]] : i32
func.func @diff_op(%mask: i1, %a: i32, %b: i32, %k: i32) -> i32 {
  %x = arith.muli %a, %k overflow<nsw> : i32
  %y = arith.muli %b, %k overflow<nuw> : i32
  %r = arith.select %mask, %x, %y : i32
  return %r : i32
}

// CHECK-LABEL: func.func @non_elementwise(
// CHECK-NEXT: %[[X:.*]] = vector.shuffle %arg1, %arg3 [1, 4, 3, 6] : vector<4xf32>, vector<4xf32>
// CHECK-NEXT: %[[Y:.*]] = vector.shuffle %arg2, %arg3 [1, 4, 3, 6] : vector<4xf32>, vector<4xf32>
// CHECK-NEXT: %[[R:.*]] = arith.select %arg0, %[[X]], %[[Y]] : vector<4xi1>, vector<4xf32>
// CHECK-NEXT: return %[[R]] : vector<4xf32>
func.func @non_elementwise(%mask: vector<4xi1>, %a: vector<4xf32>, %b: vector<4xf32>, %k: vector<4xf32>) -> vector<4xf32> {
  %x = vector.shuffle %a, %k [1, 4, 3, 6] : vector<4xf32>, vector<4xf32>
  %y = vector.shuffle %b, %k [1, 4, 3, 6] : vector<4xf32>, vector<4xf32>
  %r = arith.select %mask, %x, %y : vector<4xi1>, vector<4xf32>
  return %r : vector<4xf32>
}

// If the regions capture different values, do not transform.
// CHECK-LABEL: func.func @region_diff(
// CHECK-NEXT: %[[X:.*]] = "test.pure_elementwise_region"(%arg1) ({
// CHECK-NEXT: ^bb0(%[[A:.*]]: f32):
// CHECK-NEXT: %[[SUM0:.*]] = arith.addf %[[A]], %arg3 : f32
// CHECK-NEXT: "test.return"(%[[SUM0]]) : (f32) -> ()
// CHECK-NEXT: }) : (f32) -> f32
// CHECK-NEXT: %[[Y:.*]] = "test.pure_elementwise_region"(%arg2) ({
// CHECK-NEXT: ^bb0(%[[B:.*]]: f32):
// CHECK-NEXT: %[[SUM1:.*]] = arith.addf %[[B]], %arg4 : f32
// CHECK-NEXT: "test.return"(%[[SUM1]]) : (f32) -> ()
// CHECK-NEXT: }) : (f32) -> f32
// CHECK-NEXT: %[[R:.*]] = arith.select %arg0, %[[X]], %[[Y]] : f32
// CHECK-NEXT: return %[[R]] : f32
func.func @region_diff(%mask: i1, %a: f32, %b: f32, %k0: f32, %k1: f32) -> f32 {
  %x = "test.pure_elementwise_region"(%a) ({
  ^bb0(%v: f32):
    %sum = arith.addf %v, %k0 : f32
    "test.return"(%sum) : (f32) -> ()
  }) : (f32) -> f32
  %y = "test.pure_elementwise_region"(%b) ({
  ^bb0(%v: f32):
    %sum = arith.addf %v, %k1 : f32
    "test.return"(%sum) : (f32) -> ()
  }) : (f32) -> f32
  %r = arith.select %mask, %x, %y : f32
  return %r : f32
}

// The transformation is meant to be valid but our current implementation
// limitations can't handle the case properly.
// CHECK-LABEL: func.func @region_same(
// CHECK-NEXT: %[[X:.*]] = "test.pure_elementwise_region"(%arg1) ({
// CHECK-NEXT: ^bb0(%[[A:.*]]: f32):
// CHECK-NEXT: %[[SUM0:.*]] = arith.addf %[[A]], %arg3 : f32
// CHECK-NEXT: "test.return"(%[[SUM0]]) : (f32) -> ()
// CHECK-NEXT: }) : (f32) -> f32
// CHECK-NEXT: %[[Y:.*]] = "test.pure_elementwise_region"(%arg2) ({
// CHECK-NEXT: ^bb0(%[[B:.*]]: f32):
// CHECK-NEXT: %[[SUM1:.*]] = arith.addf %[[B]], %arg3 : f32
// CHECK-NEXT: "test.return"(%[[SUM1]]) : (f32) -> ()
// CHECK-NEXT: }) : (f32) -> f32
// CHECK-NEXT: %[[R:.*]] = arith.select %arg0, %[[X]], %[[Y]] : f32
// CHECK-NEXT: return %[[R]] : f32
func.func @region_same(%mask: i1, %a: f32, %b: f32, %k0: f32) -> f32 {
  %x = "test.pure_elementwise_region"(%a) ({
  ^bb0(%v: f32):
    %sum = arith.addf %v, %k0 : f32
    "test.return"(%sum) : (f32) -> ()
  }) : (f32) -> f32
  %y = "test.pure_elementwise_region"(%b) ({
  ^bb0(%v: f32):
    %sum = arith.addf %v, %k0 : f32
    "test.return"(%sum) : (f32) -> ()
  }) : (f32) -> f32
  %r = arith.select %mask, %x, %y : f32
  return %r : f32
}

// If F is outside the loop, do not move it into the loop.
// CHECK-LABEL: func.func @outside_loop(
// CHECK-NEXT: %[[X:.*]] = math.sqrt %arg4 : f32
// CHECK-NEXT: %[[Y:.*]] = math.sqrt %arg5 : f32
// CHECK-NEXT: %[[R:.*]] = scf.for %[[I:.*]] = %arg0 to %arg1 step %arg2 iter_args(%[[ACC:.*]] = %arg6) -> (f32) {
// CHECK-NEXT: %[[MASK:.*]] = arith.cmpi slt, %[[I]], %arg3 : index
// CHECK-NEXT: %[[S:.*]] = arith.select %[[MASK]], %[[X]], %[[Y]] : f32
// CHECK-NEXT: %[[SUM:.*]] = arith.addf %[[ACC]], %[[S]] : f32
// CHECK-NEXT: scf.yield %[[SUM]] : f32
// CHECK-NEXT: }
// CHECK-NEXT: return %[[R]] : f32
func.func @outside_loop(%lb: index, %ub: index, %step: index, %split: index, %a: f32, %b: f32, %init: f32) -> f32 {
  %x = math.sqrt %a : f32
  %y = math.sqrt %b : f32
  %r = scf.for %i = %lb to %ub step %step iter_args(%acc = %init) -> f32 {
    %mask = arith.cmpi slt, %i, %split : index
    %s = arith.select %mask, %x, %y : f32
    %sum = arith.addf %acc, %s : f32
    scf.yield %sum : f32
  }
  return %r : f32
}

// Repeated operands share one select; a common operand needs no select.
// CHECK-LABEL: func.func @repeated_operands(
// CHECK-NEXT: %[[S:.*]] = arith.select %arg0, %arg1, %arg2 : f32
// CHECK-NEXT: %[[R:.*]] = math.fma %[[S]], %[[S]], %arg3 : f32
// CHECK-NEXT: return %[[R]] : f32
func.func @repeated_operands(%mask: i1, %a: f32, %b: f32, %k: f32) -> f32 {
  %x = math.fma %a, %a, %k : f32
  %y = math.fma %b, %b, %k : f32
  %r = arith.select %mask, %x, %y : f32
  return %r : f32
}

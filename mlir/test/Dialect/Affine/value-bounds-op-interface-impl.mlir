// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(test-affine-reify-value-bounds))' -verify-diagnostics \
// RUN:     -split-input-file | FileCheck %s

// CHECK: #[[$map:.*]] = affine_map<()[s0, s1] -> (s0 + s1)>
// CHECK-LABEL: func @affine_apply(
//  CHECK-SAME:     %[[a:.*]]: index, %[[b:.*]]: index
//       CHECK:   %[[apply:.*]] = affine.apply #[[$map]]()[%[[a]], %[[b]]]
//       CHECK:   %[[apply:.*]] = affine.apply #[[$map]]()[%[[a]], %[[b]]]
//       CHECK:   return %[[apply]]
func.func @affine_apply(%a: index, %b: index) -> index {
  %0 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%a, %b]
  %1 = "test.reify_bound"(%0) : (index) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @affine_max_lb(
//  CHECK-SAME:     %[[a:.*]]: index
//       CHECK:   %[[c2:.*]] = arith.constant 2 : index
//       CHECK:   return %[[c2]]
func.func @affine_max_lb(%a: index) -> (index) {
  // Note: There are two LBs: s0 and 2. FlatAffineValueConstraints always
  // returns the constant one at the moment.
  %1 = affine.max affine_map<()[s0] -> (s0, 2)>()[%a]
  %2 = "test.reify_bound"(%1) {type = "LB"}: (index) -> (index)
  return %2 : index
}

// -----

func.func @affine_max_ub(%a: index) -> (index) {
  %1 = affine.max affine_map<()[s0] -> (s0, 2)>()[%a]
  // expected-error @below{{could not reify bound}}
  %2 = "test.reify_bound"(%1) {type = "UB"}: (index) -> (index)
  return %2 : index
}

// -----

// CHECK-LABEL: func @affine_min_ub(
//  CHECK-SAME:     %[[a:.*]]: index
//       CHECK:   %[[c3:.*]] = arith.constant 3 : index
//       CHECK:   return %[[c3]]
func.func @affine_min_ub(%a: index) -> (index) {
  // Note: There are two UBs: s0 + 1 and 3. FlatAffineValueConstraints always
  // returns the constant one at the moment.
  %1 = affine.min affine_map<()[s0] -> (s0, 2)>()[%a]
  %2 = "test.reify_bound"(%1) {type = "UB"}: (index) -> (index)
  return %2 : index
}

// -----

func.func @affine_min_lb(%a: index) -> (index) {
  %1 = affine.min affine_map<()[s0] -> (s0, 2)>()[%a]
  // expected-error @below{{could not reify bound}}
  %2 = "test.reify_bound"(%1) {type = "LB"}: (index) -> (index)
  return %2 : index
}

// -----

// CHECK-LABEL: func @composed_affine_apply(
//       CHECK:   %[[cst:.*]] = arith.constant -8 : index
//       CHECK:   return %[[cst]]
func.func @composed_affine_apply(%i1 : index) -> (index) {
  // The ValueBoundsOpInterface implementation of affine.apply fully composes
  // the affine map (and its operands) with other affine.apply ops drawn from
  // its operands before adding it to the constraint set. This is to work
  // around a limitation in `FlatLinearConstraints`, which can currently not
  // compute a constant bound for %s. (The affine map simplification logic can
  // simplify %s to -8.)
  %i2 = affine.apply affine_map<(d0) -> ((d0 floordiv 32) * 16)>(%i1)
  %i3 = affine.apply affine_map<(d0) -> ((d0 floordiv 32) * 16 + 8)>(%i1)
  %s = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%i2, %i3]
  %reified = "test.reify_bound"(%s) {type = "EQ", constant} : (index) -> (index)
  return %reified : index
}


// -----

func.func @are_equal(%i1 : index) {
  %i2 = affine.apply affine_map<(d0) -> ((d0 floordiv 32) * 16)>(%i1)
  %i3 = affine.apply affine_map<(d0) -> ((d0 floordiv 32) * 16 + 8)>(%i1)
  %s = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%i2, %i3]
  // expected-remark @below{{false}}
   "test.compare"(%i2, %i3) : (index, index) -> ()
  return
}

// -----

// Test for affine::fullyComposeAndCheckIfEqual
func.func @composed_are_equal(%i1 : index) {
  %i2 = affine.apply affine_map<(d0) -> ((d0 floordiv 32) * 16)>(%i1)
  %i3 = affine.apply affine_map<(d0) -> ((d0 floordiv 32) * 16 + 8)>(%i1)
  %s = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%i2, %i3]
  // expected-remark @below{{different}}
   "test.compare"(%i2, %i3) {compose} : (index, index) -> ()
  return
}

// -----

func.func @compare_affine_max(%a: index, %b: index) {
  %0 = affine.max affine_map<()[s0, s1] -> (s0, s1)>()[%a, %b]
  // expected-remark @below{{true}}
  "test.compare"(%0, %a) {cmp = "GE"} : (index, index) -> ()
  // expected-error @below{{unknown}}
  "test.compare"(%0, %a) {cmp = "GT"} : (index, index) -> ()
  // expected-remark @below{{false}}
  "test.compare"(%0, %a) {cmp = "LT"} : (index, index) -> ()
  // expected-error @below{{unknown}}
  "test.compare"(%0, %a) {cmp = "LE"} : (index, index) -> ()
  return
}

// -----

func.func @compare_affine_min(%a: index, %b: index) {
  %0 = affine.min affine_map<()[s0, s1] -> (s0, s1)>()[%a, %b]
  // expected-error @below{{unknown}}
  "test.compare"(%0, %a) {cmp = "GE"} : (index, index) -> ()
  // expected-remark @below{{false}}
  "test.compare"(%0, %a) {cmp = "GT"} : (index, index) -> ()
  // expected-error @below{{unknown}}
  "test.compare"(%0, %a) {cmp = "LT"} : (index, index) -> ()
  // expected-remark @below{{true}}
  "test.compare"(%0, %a) {cmp = "LE"} : (index, index) -> ()
  return
}

// -----

func.func @compare_const_map() {
  %c5 = arith.constant 5 : index
  // expected-remark @below{{true}}
  "test.compare"(%c5) {cmp = "GT", rhs_map = affine_map<() -> (4)>}
      : (index) -> ()
  // expected-remark @below{{true}}
  "test.compare"(%c5) {cmp = "LT", lhs_map = affine_map<() -> (4)>}
      : (index) -> ()
  return
}

// -----

func.func @compare_maps(%a: index, %b: index) {
  // expected-remark @below{{true}}
  "test.compare"(%a, %b, %b, %a)
      {cmp = "GT", lhs_map = affine_map<(d0, d1) -> (1 + d0 + d1)>,
       rhs_map = affine_map<(d0, d1) -> (d0 + d1)>}
      : (index, index, index, index) -> ()
  return
}

// -----

// CHECK-DAG: #[[$map1:.+]] = affine_map<()[s0] -> (s0 floordiv 15)>
// CHECK-DAG: #[[$map2:.+]] = affine_map<()[s0] -> ((s0 mod 15) floordiv 5)>
// CHECK-DAG: #[[$map3:.+]] = affine_map<()[s0] -> (s0 mod 5)>
// CHECK-LABEL: func.func @delinearize_static
// CHECK-SAME: (%[[arg0:.+]]: index)
// CHECK-DAG: %[[v1:.+]] = affine.apply #[[$map1]]()[%[[arg0]]]
// CHECK-DAG: %[[v2:.+]] = affine.apply #[[$map2]]()[%[[arg0]]]
// CHECK-DAG: %[[v3:.+]] = affine.apply #[[$map3]]()[%[[arg0]]]
// CHECK: return %[[v1]], %[[v2]], %[[v3]]
func.func @delinearize_static(%arg0: index) -> (index, index, index) {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %0:3 = affine.delinearize_index %arg0 into (2, 3, 5) : index, index, index
  %1 = "test.reify_bound"(%0#0) {type = "EQ"} : (index) -> (index)
  %2 = "test.reify_bound"(%0#1) {type = "EQ"} : (index) -> (index)
  %3 = "test.reify_bound"(%0#2) {type = "EQ"} : (index) -> (index)
  // expected-remark @below{{true}}
  "test.compare"(%0#0, %c2) {cmp = "LT"} : (index, index) -> ()
  // expected-remark @below{{true}}
  "test.compare"(%0#1, %c3) {cmp = "LT"} : (index, index) -> ()
  return %1, %2, %3 : index, index, index
}

// -----

// CHECK-DAG: #[[$map1:.+]] = affine_map<()[s0] -> (s0 floordiv 15)>
// CHECK-DAG: #[[$map2:.+]] = affine_map<()[s0] -> ((s0 mod 15) floordiv 5)>
// CHECK-DAG: #[[$map3:.+]] = affine_map<()[s0] -> (s0 mod 5)>
// CHECK-LABEL: func.func @delinearize_static_no_outer_bound
// CHECK-SAME: (%[[arg0:.+]]: index)
// CHECK-DAG: %[[v1:.+]] = affine.apply #[[$map1]]()[%[[arg0]]]
// CHECK-DAG: %[[v2:.+]] = affine.apply #[[$map2]]()[%[[arg0]]]
// CHECK-DAG: %[[v3:.+]] = affine.apply #[[$map3]]()[%[[arg0]]]
// CHECK: return %[[v1]], %[[v2]], %[[v3]]
func.func @delinearize_static_no_outer_bound(%arg0: index) -> (index, index, index) {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %0:3 = affine.delinearize_index %arg0 into (3, 5) : index, index, index
  %1 = "test.reify_bound"(%0#0) {type = "EQ"} : (index) -> (index)
  %2 = "test.reify_bound"(%0#1) {type = "EQ"} : (index) -> (index)
  %3 = "test.reify_bound"(%0#2) {type = "EQ"} : (index) -> (index)
  "test.compaare"(%0#0, %c2) {cmp = "LT"} : (index, index) -> ()
  // expected-remark @below{{true}}
  "test.compare"(%0#1, %c3) {cmp = "LT"} : (index, index) -> ()
  return %1, %2, %3 : index, index, index
}

// -----

// CHECK: #[[$map:.+]] = affine_map<()[s0, s1] -> (s0 + s1 * 3)>
// CHECK-LABEL: func.func @linearize_static
// CHECK-SAME: (%[[arg0:.+]]: index, %[[arg1:.+]]: index)
// CHECK: %[[v1:.+]] = affine.apply #[[$map]]()[%[[arg1]], %[[arg0]]]
// CHECK: return %[[v1]]
func.func @linearize_static(%arg0: index, %arg1: index)  -> index {
  %c6 = arith.constant 6 : index
  %0 = affine.linearize_index disjoint [%arg0, %arg1] by (2, 3) : index
  %1 = "test.reify_bound"(%0) {type = "EQ"} : (index) -> (index)
  // expected-remark @below{{true}}
  "test.compare"(%0, %c6) {cmp = "LT"} : (index, index) -> ()
  return %1 : index
}

// -----

// CHECK: #[[$map:.+]] = affine_map<()[s0, s1] -> (s0 + s1 * 3)>
// CHECK-LABEL: func.func @linearize_static_no_outer_bound
// CHECK-SAME: (%[[arg0:.+]]: index, %[[arg1:.+]]: index)
// CHECK: %[[v1:.+]] = affine.apply #[[$map]]()[%[[arg1]], %[[arg0]]]
// CHECK: return %[[v1]]
func.func @linearize_static_no_outer_bound(%arg0: index, %arg1: index)  -> index {
  %c6 = arith.constant 6 : index
  %0 = affine.linearize_index disjoint [%arg0, %arg1] by (3) : index
  %1 = "test.reify_bound"(%0) {type = "EQ"} : (index) -> (index)
  // expected-error @below{{unknown}}
  "test.compare"(%0, %c6) {cmp = "LT"} : (index, index) -> ()
  return %1 : index
}

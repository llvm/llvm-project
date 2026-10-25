// RUN: mlir-opt %s -canonicalize='test-convergence' -split-input-file | FileCheck %s

// Branch-local uses can be simplified even when the select must remain for an
// outside use. Repeated operands of the same operation are all replaced.
// CHECK-LABEL: func.func @captured_select(
// CHECK-SAME: %[[P:.*]]: i1, %[[A:.*]]: i32, %[[B:.*]]: i32
// CHECK: %[[X:.*]] = arith.select %[[P]], %[[A]], %[[B]] : i32
// CHECK: scf.if %[[P]] {
// CHECK: func.call @consume_pair(%[[A]], %[[A]])
// CHECK: } else {
// CHECK: func.call @consume_pair(%[[B]], %[[B]])
// CHECK: return %[[X]] : i32
func.func @captured_select(%p: i1, %a: i32, %b: i32) -> i32 {
  %x = arith.select %p, %a, %b : i32
  scf.if %p {
    func.call @consume_pair(%x, %x) : (i32, i32) -> ()
  } else {
    func.call @consume_pair(%x, %x) : (i32, i32) -> ()
  }
  return %x : i32
}
func.func private @consume_pair(i32, i32)

// -----

// CHECK-LABEL: func.func @then_only(
// CHECK-SAME: %[[P:.*]]: i1, %[[A:.*]]: i32, %[[B:.*]]: i32
// CHECK-NOT: arith.select
// CHECK: scf.if %[[P]] {
// CHECK: func.call @consume(%[[A]])
// CHECK-NOT: arith.select
// CHECK: return
func.func @then_only(%p: i1, %a: i32, %b: i32) {
  %x = arith.select %p, %a, %b : i32
  scf.if %p {
    func.call @consume(%x) : (i32) -> ()
  }
  return
}
func.func private @consume(i32)

// -----

// Selecting a memref does not copy its contents. Stores stay in their original
// branches and use the selected destination; outside uses keep the select.
// CHECK-LABEL: func.func @memref_destination(
// CHECK-SAME: %[[P:.*]]: i1, %[[A:.*]]: memref<16xi32>, %[[B:.*]]: memref<16xi32>
// CHECK: %[[X:.*]] = arith.select %[[P]], %[[A]], %[[B]] : memref<16xi32>
// CHECK: scf.if %[[P]] {
// CHECK: memref.store %{{.*}}, %[[A]][%{{.*}}]
// CHECK: } else {
// CHECK: memref.store %{{.*}}, %[[B]][%{{.*}}]
// CHECK: return %[[X]] : memref<16xi32>
func.func @memref_destination(%p: i1, %a: memref<16xi32>, %b: memref<16xi32>) -> memref<16xi32> {
  %c0 = arith.constant 0 : index
  %v = arith.constant 42 : i32
  %x = arith.select %p, %a, %b : memref<16xi32>
  scf.if %p {
    memref.store %v, %x[%c0] : memref<16xi32>
  } else {
    memref.store %v, %x[%c0] : memref<16xi32>
  }
  return %x : memref<16xi32>
}

// -----

// The outer branch's condition also applies to uses in nested regions.
// CHECK-LABEL: func.func @nested_regions(
// CHECK-SAME: %[[P:.*]]: i1, %[[Q:.*]]: i1, %[[A:.*]]: i32, %[[B:.*]]: i32
// CHECK-NOT: arith.select
// CHECK: scf.if %[[P]] {
// CHECK: scf.if %[[Q]] {
// CHECK: func.call @consume(%[[A]])
// CHECK: } else {
// CHECK: func.call @consume(%[[A]])
// CHECK: } else {
// CHECK: func.call @consume(%[[B]])
func.func @nested_regions(%p: i1, %q: i1, %a: i32, %b: i32) {
  %x = arith.select %p, %a, %b : i32
  scf.if %p {
    scf.if %q {
      func.call @consume(%x) : (i32) -> ()
    } else {
      func.call @consume(%x) : (i32) -> ()
    }
  } else {
    func.call @consume(%x) : (i32) -> ()
  }
  return
}
func.func private @consume(i32)

// -----

// CHECK-LABEL: func.func @loop_region(
// CHECK-SAME: %[[P:.*]]: i1, %[[A:.*]]: i32, %[[B:.*]]: i32, %[[N:.*]]: index
// CHECK-NOT: arith.select
// CHECK: scf.if %[[P]] {
// CHECK: scf.for
// CHECK: func.call @consume(%[[A]])
// CHECK: } else {
// CHECK: func.call @consume(%[[B]])
func.func @loop_region(%p: i1, %a: i32, %b: i32, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %x = arith.select %p, %a, %b : i32
  scf.if %p {
    scf.for %i = %c0 to %n step %c1 {
      func.call @consume(%x) : (i32) -> ()
    }
  } else {
    func.call @consume(%x) : (i32) -> ()
  }
  return
}
func.func private @consume(i32)

// -----

// A branch on an unrelated condition cannot refine the selected value.
// CHECK-LABEL: func.func @unrelated_condition(
// CHECK-SAME: %[[P:.*]]: i1, %[[Q:.*]]: i1, %[[A:.*]]: i32, %[[B:.*]]: i32
// CHECK: %[[X:.*]] = arith.select %[[P]], %[[A]], %[[B]] : i32
// CHECK: scf.if %[[Q]] {
// CHECK: func.call @consume(%[[X]])
// CHECK: } else {
// CHECK: func.call @consume(%[[X]])
func.func @unrelated_condition(%p: i1, %q: i1, %a: i32, %b: i32) {
  %x = arith.select %p, %a, %b : i32
  scf.if %q {
    func.call @consume(%x) : (i32) -> ()
  } else {
    func.call @consume(%x) : (i32) -> ()
  }
  return
}
func.func private @consume(i32)

// -----

// Merely using the branch condition as a select arm is not sufficient.
// CHECK-LABEL: func.func @condition_as_select_arm(
// CHECK-SAME: %[[P:.*]]: i1, %[[Q:.*]]: i1, %[[B:.*]]: i1
// CHECK: %[[X:.*]] = arith.select %[[Q]], %[[P]], %[[B]] : i1
// CHECK: scf.if %[[P]] {
// CHECK: func.call @consume_i1(%[[X]])
// CHECK: } else {
// CHECK: func.call @consume_i1(%[[X]])
func.func @condition_as_select_arm(%p: i1, %q: i1, %b: i1) {
  %x = arith.select %q, %p, %b : i1
  scf.if %p {
    func.call @consume_i1(%x) : (i1) -> ()
  } else {
    func.call @consume_i1(%x) : (i1) -> ()
  }
  return
}
func.func private @consume_i1(i1)

// -----

// Canonicalization of a negated if condition must preserve which arm is used.
// CHECK-LABEL: func.func @negated_condition(
// CHECK-SAME: %[[P:.*]]: i1, %[[A:.*]]: i32, %[[B:.*]]: i32
// CHECK-NOT: arith.select
// CHECK: scf.if %[[P]] {
// CHECK: func.call @consume_pair(%[[A]], %[[B]])
// CHECK: } else {
// CHECK: func.call @consume_pair(%[[B]], %[[A]])
func.func @negated_condition(%p: i1, %a: i32, %b: i32) {
  %true = arith.constant true
  %not_p = arith.xori %p, %true : i1
  %x = arith.select %p, %a, %b : i32
  scf.if %not_p {
    func.call @consume_pair(%x, %a) : (i32, i32) -> ()
  } else {
    func.call @consume_pair(%x, %b) : (i32, i32) -> ()
  }
  return
}
func.func private @consume_pair(i32, i32)

// -----

// A scalar branch condition does not refine an elementwise vector condition.
// CHECK-LABEL: func.func @vector_condition(
// CHECK-SAME: %[[P:.*]]: i1, %[[MASK:.*]]: vector<4xi1>, %[[A:.*]]: vector<4xi32>, %[[B:.*]]: vector<4xi32>
// CHECK: %[[X:.*]] = arith.select %[[MASK]], %[[A]], %[[B]]
// CHECK: scf.if %[[P]] {
// CHECK: func.call @consume_vector(%[[X]])
// CHECK: } else {
// CHECK: func.call @consume_vector(%[[X]])
func.func @vector_condition(%p: i1, %mask: vector<4xi1>, %a: vector<4xi32>, %b: vector<4xi32>) {
  %x = arith.select %mask, %a, %b : vector<4xi1>, vector<4xi32>
  scf.if %p {
    func.call @consume_vector(%x) : (vector<4xi32>) -> ()
  } else {
    func.call @consume_vector(%x) : (vector<4xi32>) -> ()
  }
  return
}
func.func private @consume_vector(vector<4xi32>)

// -----

// A scalar select can select an entire vector without changing its elements.
// CHECK-LABEL: func.func @scalar_condition_vector_result(
// CHECK-SAME: %[[P:.*]]: i1, %[[A:.*]]: vector<4xi32>, %[[B:.*]]: vector<4xi32>
// CHECK-NOT: arith.select
// CHECK: scf.if %[[P]] {
// CHECK: func.call @consume_vector(%[[A]])
// CHECK: } else {
// CHECK: func.call @consume_vector(%[[B]])
func.func @scalar_condition_vector_result(%p: i1, %a: vector<4xi32>, %b: vector<4xi32>) {
  %x = arith.select %p, %a, %b : vector<4xi32>
  scf.if %p {
    func.call @consume_vector(%x) : (vector<4xi32>) -> ()
  } else {
    func.call @consume_vector(%x) : (vector<4xi32>) -> ()
  }
  return
}
func.func private @consume_vector(vector<4xi32>)

// -----

// An isolated region receives the selected value as an operand. Its block
// argument remains local to that region; it must not capture either select arm.
// CHECK-LABEL: func.func @isolated_region_operand(
// CHECK-SAME: %[[P:.*]]: i1, %[[A:.*]]: index, %[[B:.*]]: index
// CHECK-NOT: arith.select
// CHECK: scf.if %[[P]] {
// CHECK: test.isolated_region %[[A]] {
// CHECK: "test.consume"(%[[A]]) : (index) -> ()
// CHECK: } else {
// CHECK: test.isolated_region %[[B]] {
// CHECK: "test.consume"(%[[B]]) : (index) -> ()
func.func @isolated_region_operand(%p: i1, %a: index, %b: index) {
  %x = arith.select %p, %a, %b : index
  scf.if %p {
    test.isolated_region %x {
      "test.consume"(%x) : (index) -> ()
    }
  } else {
    test.isolated_region %x {
      "test.consume"(%x) : (index) -> ()
    }
  }
  return
}

// -----

// Correlated selects may form a chain. Both branch-local operands are refined,
// while the result returned outside the if still needs a select.
// CHECK-LABEL: func.func @select_chain(
// CHECK-SAME: %[[P:.*]]: i1, %[[A:.*]]: i32, %[[B:.*]]: i32, %[[C:.*]]: i32
// CHECK: %[[RESULT:.*]] = arith.select %[[P]], %[[C]], %[[B]] : i32
// CHECK: scf.if %[[P]] {
// CHECK: func.call @consume_pair(%[[A]], %[[C]])
// CHECK: } else {
// CHECK: func.call @consume_pair(%[[B]], %[[B]])
// CHECK: return %[[RESULT]] : i32
func.func @select_chain(%p: i1, %a: i32, %b: i32, %c: i32) -> i32 {
  %x = arith.select %p, %a, %b : i32
  %y = arith.select %p, %c, %x : i32
  scf.if %p {
    func.call @consume_pair(%x, %y) : (i32, i32) -> ()
  } else {
    func.call @consume_pair(%x, %y) : (i32, i32) -> ()
  }
  return %y : i32
}
func.func private @consume_pair(i32, i32)

// RUN: mlir-opt --allow-unregistered-dialect %s \
// RUN:   --pass-pipeline='builtin.module(func.func(affine-loop-carried-computation-reuse),canonicalize,cse)' \
// RUN:   | FileCheck %s

// A non-trivial producer DAG at i+1 is identical to the DAG at i in the next
// iteration. Carry its result without interpreting addi or muli as reductions.

// CHECK-LABEL: func.func @integer_polynomial
// CHECK: %[[A0:.*]] = affine.load %[[SRC:.*]][0]
// CHECK: %[[A02:.*]] = arith.muli %[[A0]], %[[A0]]
// CHECK: %[[B0:.*]] = affine.load %[[SRC]][1]
// CHECK: %[[INITIAL:.*]] = arith.addi %[[A02]], %[[B0]]
// CHECK: affine.for %[[I:.*]] = 0 to 8 iter_args(%[[PREVIOUS:.*]] = %[[INITIAL]])
// CHECK-NOT: affine.load %[[SRC]][%[[I]]]
// CHECK: %[[B:.*]] = affine.load %[[SRC]][%[[I]] + 1]
// CHECK: %[[C:.*]] = affine.load %[[SRC]][%[[I]] + 2]
// CHECK: %[[B2:.*]] = arith.muli %[[B]], %[[B]]
// CHECK: %[[CURRENT:.*]] = arith.addi %[[B2]], %[[C]]
// CHECK: arith.subi %[[CURRENT]], %[[PREVIOUS]]
// CHECK: affine.yield %[[CURRENT]]
func.func @integer_polynomial(%src0: memref<10xi32>,
                              %dst0: memref<8xi32>) {
  %src, %dst = memref.distinct_objects %src0, %dst0
      : memref<10xi32>, memref<8xi32>
  affine.for %i = 0 to 8 {
    %a = affine.load %src[%i] : memref<10xi32>
    %b = affine.load %src[%i + 1] : memref<10xi32>
    %c = affine.load %src[%i + 2] : memref<10xi32>
    %a2 = arith.muli %a, %a : i32
    %left = arith.addi %a2, %b : i32
    %b2 = arith.muli %b, %b : i32
    %right = arith.addi %b2, %c : i32
    %difference = arith.subi %right, %left : i32
    affine.store %difference, %dst[%i] : memref<8xi32>
  }
  return
}

// Existing ordered and reduction state keeps its original position. The
// translated producer state is appended.

// CHECK-LABEL: func.func @existing_iter_arg
// CHECK: %[[RESULT:.*]]:2 = affine.for %[[I:.*]] = 0 to 8
// CHECK-SAME: iter_args(%[[ACC:.*]] = %{{.*}}, %[[PREVIOUS:.*]] = %{{.*}})
// CHECK: %[[CURRENT:.*]] = arith.addi
// CHECK: %[[NEXT:.*]] = arith.addi %[[ACC]], %{{.*}}
// CHECK: affine.yield %[[NEXT]], %[[CURRENT]]
// CHECK: return %[[RESULT]]#0
func.func @existing_iter_arg(%src0: memref<10xi32>,
                             %dst0: memref<8xi32>) -> i32 {
  %src, %dst = memref.distinct_objects %src0, %dst0
      : memref<10xi32>, memref<8xi32>
  %zero = arith.constant 0 : i32
  %result = affine.for %i = 0 to 8 iter_args(%acc = %zero) -> i32 {
    %a = affine.load %src[%i] : memref<10xi32>
    %b = affine.load %src[%i + 1] : memref<10xi32>
    %c = affine.load %src[%i + 2] : memref<10xi32>
    %left = arith.addi %a, %b : i32
    %right = arith.addi %b, %c : i32
    %difference = arith.subi %right, %left : i32
    affine.store %difference, %dst[%i] : memref<8xi32>
    %next = arith.addi %acc, %difference : i32
    affine.yield %next : i32
  }
  return %result : i32
}

// A raw-load pair is intentionally outside this computation-reuse pass.

// CHECK-LABEL: func.func @raw_load_only
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @raw_load_only(%src0: memref<9xi32>, %dst0: memref<8xi32>) {
  %src, %dst = memref.distinct_objects %src0, %dst0
      : memref<9xi32>, memref<8xi32>
  affine.for %i = 0 to 8 {
    %left = affine.load %src[%i] : memref<9xi32>
    %right = affine.load %src[%i + 1] : memref<9xi32>
    %difference = arith.subi %right, %left : i32
    affine.store %difference, %dst[%i] : memref<8xi32>
  }
  return
}

// Without object disjointness, the destination write may modify the source
// before the carried value is consumed in the next iteration.

// CHECK-LABEL: func.func @may_alias_output
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @may_alias_output(%src: memref<10xi32>, %dst: memref<8xi32>) {
  affine.for %i = 0 to 8 {
    %a = affine.load %src[%i] : memref<10xi32>
    %b = affine.load %src[%i + 1] : memref<10xi32>
    %c = affine.load %src[%i + 2] : memref<10xi32>
    %left = arith.addi %a, %b : i32
    %right = arith.addi %b, %c : i32
    %difference = arith.subi %right, %left : i32
    affine.store %difference, %dst[%i] : memref<8xi32>
  }
  return
}

// A write to the producer source invalidates cross-iteration reuse.

// CHECK-LABEL: func.func @source_modified
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @source_modified(%src0: memref<10xi32>, %dst0: memref<8xi32>) {
  %src, %dst = memref.distinct_objects %src0, %dst0
      : memref<10xi32>, memref<8xi32>
  affine.for %i = 0 to 8 {
    %a = affine.load %src[%i] : memref<10xi32>
    %b = affine.load %src[%i + 1] : memref<10xi32>
    %c = affine.load %src[%i + 2] : memref<10xi32>
    %left = arith.addi %a, %b : i32
    %right = arith.addi %b, %c : i32
    affine.store %right, %src[%i + 1] : memref<10xi32>
    %difference = arith.subi %right, %left : i32
    affine.store %difference, %dst[%i] : memref<8xi32>
  }
  return
}

// Structurally different roots and a non-unit translation do not match.

// CHECK-LABEL: func.func @different_operator
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @different_operator(%src0: memref<10xi32>,
                              %dst0: memref<8xi32>) {
  %src, %dst = memref.distinct_objects %src0, %dst0
      : memref<10xi32>, memref<8xi32>
  affine.for %i = 0 to 8 {
    %a = affine.load %src[%i] : memref<10xi32>
    %b = affine.load %src[%i + 1] : memref<10xi32>
    %c = affine.load %src[%i + 2] : memref<10xi32>
    %left = arith.addi %a, %b : i32
    %right = arith.muli %b, %c : i32
    %difference = arith.subi %right, %left : i32
    affine.store %difference, %dst[%i] : memref<8xi32>
  }
  return
}

// CHECK-LABEL: func.func @distance_two
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @distance_two(%src0: memref<11xi32>, %dst0: memref<8xi32>) {
  %src, %dst = memref.distinct_objects %src0, %dst0
      : memref<11xi32>, memref<8xi32>
  affine.for %i = 0 to 8 {
    %a = affine.load %src[%i] : memref<11xi32>
    %b = affine.load %src[%i + 1] : memref<11xi32>
    %c = affine.load %src[%i + 3] : memref<11xi32>
    %left = arith.addi %a, %b : i32
    %right = arith.addi %b, %c : i32
    %difference = arith.subi %right, %left : i32
    affine.store %difference, %dst[%i] : memref<8xi32>
  }
  return
}

// Hoisting the initial producer out of a loop that may not execute would add
// memory reads and arithmetic.

// CHECK-LABEL: func.func @dynamic_trip
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @dynamic_trip(%src0: memref<?xi32>, %dst0: memref<?xi32>,
                        %n: index) {
  %src, %dst = memref.distinct_objects %src0, %dst0
      : memref<?xi32>, memref<?xi32>
  affine.for %i = 0 to %n {
    %a = affine.load %src[%i] : memref<?xi32>
    %b = affine.load %src[%i + 1] : memref<?xi32>
    %c = affine.load %src[%i + 2] : memref<?xi32>
    %left = arith.addi %a, %b : i32
    %right = arith.addi %b, %c : i32
    %difference = arith.subi %right, %left : i32
    affine.store %difference, %dst[%i] : memref<?xi32>
  }
  return
}

// A producer depending on another loop-carried value has a different context
// key in the next iteration and must not be reused.

// CHECK-LABEL: func.func @context_changes
// CHECK: affine.for
// CHECK-NOT: iter_args({{.*}},
func.func @context_changes(%src0: memref<10xi32>, %dst0: memref<8xi32>) {
  %src, %dst = memref.distinct_objects %src0, %dst0
      : memref<10xi32>, memref<8xi32>
  %zero = arith.constant 0 : i32
  affine.for %i = 0 to 8 iter_args(%state = %zero) -> i32 {
    %a = affine.load %src[%i] : memref<10xi32>
    %b = affine.load %src[%i + 1] : memref<10xi32>
    %left = arith.addi %a, %state : i32
    %right = arith.addi %b, %state : i32
    %difference = arith.subi %right, %left : i32
    affine.store %difference, %dst[%i] : memref<8xi32>
    %next = arith.addi %state, %difference : i32
    affine.yield %next : i32
  }
  return
}

// Unknown recursive effects are rejected even when the candidate loads and
// destination are otherwise distinct.

// CHECK-LABEL: func.func @unknown_effect
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @unknown_effect(%src0: memref<10xi32>, %dst0: memref<8xi32>) {
  %src, %dst = memref.distinct_objects %src0, %dst0
      : memref<10xi32>, memref<8xi32>
  affine.for %i = 0 to 8 {
    %a = affine.load %src[%i] : memref<10xi32>
    %b = affine.load %src[%i + 1] : memref<10xi32>
    %c = affine.load %src[%i + 2] : memref<10xi32>
    %left = arith.addi %a, %b : i32
    %right = arith.addi %b, %c : i32
    "test.unknown_effect"() : () -> ()
    %difference = arith.subi %right, %left : i32
    affine.store %difference, %dst[%i] : memref<8xi32>
  }
  return
}

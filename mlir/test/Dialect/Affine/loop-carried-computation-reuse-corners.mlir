// RUN: mlir-opt %s \
// RUN:   --pass-pipeline='builtin.module(func.func(affine-loop-carried-computation-reuse),canonicalize,cse)' \
// RUN:   | FileCheck %s

#lb = affine_map<(d0) -> (d0)>
#ub8 = affine_map<(d0) -> (d0 + 8)>
#ub_minus1 = affine_map<(d0) -> (d0 - 1)>
#plus0 = affine_map<(d0) -> (d0)>
#plus1 = affine_map<(d0) -> (d0 + 1)>
#plus2 = affine_map<(d0) -> (d0 + 2)>

// The translation is one iteration, not one index unit.

// CHECK-LABEL: func.func @step_two
// CHECK: %[[INIT:.*]] = arith.addi
// CHECK: affine.for %[[I:.*]] = 0 to 8 step 2 iter_args(%[[PREV:.*]] = %[[INIT]])
// CHECK: %[[CUR:.*]] = arith.addi
// CHECK: arith.subi %[[CUR]], %[[PREV]]
// CHECK: affine.yield %[[CUR]]
func.func @step_two(%src0: memref<12xi32>, %dst0: memref<4xi32>) {
  %src, %dst = memref.distinct_objects %src0, %dst0
      : memref<12xi32>, memref<4xi32>
  affine.for %i = 0 to 8 step 2 {
    %a = affine.load %src[%i] : memref<12xi32>
    %b = affine.load %src[%i + 2] : memref<12xi32>
    %c = affine.load %src[%i + 4] : memref<12xi32>
    %left = arith.addi %a, %b : i32
    %right = arith.addi %b, %c : i32
    %difference = arith.subi %right, %left : i32
    %j = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%i)
    affine.store %difference, %dst[%j] : memref<4xi32>
  }
  return
}

// CHECK-LABEL: func.func @step_mismatch
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @step_mismatch(%src0: memref<10xi32>, %dst0: memref<4xi32>) {
  %src, %dst = memref.distinct_objects %src0, %dst0
      : memref<10xi32>, memref<4xi32>
  affine.for %i = 0 to 8 step 2 {
    %a = affine.load %src[%i] : memref<10xi32>
    %b = affine.load %src[%i + 1] : memref<10xi32>
    %c = affine.load %src[%i + 2] : memref<10xi32>
    %left = arith.addi %a, %b : i32
    %right = arith.addi %b, %c : i32
    %difference = arith.subi %right, %left : i32
    %j = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%i)
    affine.store %difference, %dst[%j] : memref<4xi32>
  }
  return
}

// affine.apply chains are composed before comparing accesses.

// CHECK-LABEL: func.func @composed_accesses
// CHECK: %[[INIT:.*]] = arith.addi
// CHECK: affine.for %[[I:.*]] = 0 to 8 iter_args(%[[PREV:.*]] = %[[INIT]])
// CHECK: %[[CUR:.*]] = arith.addi
// CHECK: arith.subi %[[CUR]], %[[PREV]]
// CHECK: affine.yield %[[CUR]]
func.func @composed_accesses(%src0: memref<10xi32>, %dst0: memref<8xi32>) {
  %src, %dst = memref.distinct_objects %src0, %dst0
      : memref<10xi32>, memref<8xi32>
  affine.for %i = 0 to 8 {
    %i0 = affine.apply #plus0(%i)
    %i1 = affine.apply #plus1(%i)
    %i2 = affine.apply #plus2(%i)
    %a = affine.load %src[%i0] : memref<10xi32>
    %b = affine.load %src[%i1] : memref<10xi32>
    %c = affine.load %src[%i2] : memref<10xi32>
    %left = arith.addi %a, %b : i32
    %right = arith.addi %b, %c : i32
    %difference = arith.subi %right, %left : i32
    affine.store %difference, %dst[%i] : memref<8xi32>
  }
  return
}

// Non-linear Affine maps are accepted when canonical substitution proves the
// same one-iteration translation.

// CHECK-LABEL: func.func @modulo_accesses
// CHECK: %[[INIT:.*]] = arith.addi
// CHECK: affine.for %[[I:.*]] = 0 to 8 iter_args(%[[PREV:.*]] = %[[INIT]])
// CHECK: %[[CUR:.*]] = arith.addi
// CHECK: arith.subi %[[CUR]], %[[PREV]]
// CHECK: affine.yield %[[CUR]]
func.func @modulo_accesses(%src0: memref<4xi32>, %dst0: memref<8xi32>) {
  %src, %dst = memref.distinct_objects %src0, %dst0
      : memref<4xi32>, memref<8xi32>
  affine.for %i = 0 to 8 {
    %a = affine.load %src[%i mod 4] : memref<4xi32>
    %b = affine.load %src[(%i + 1) mod 4] : memref<4xi32>
    %c = affine.load %src[(%i + 2) mod 4] : memref<4xi32>
    %left = arith.addi %a, %b : i32
    %right = arith.addi %b, %c : i32
    %difference = arith.subi %right, %left : i32
    affine.store %difference, %dst[%i] : memref<8xi32>
  }
  return
}

// A symbolic lower bound is legal when the trip count is nevertheless known.

// CHECK-LABEL: func.func @symbolic_lower_fixed_trip
// CHECK: affine.load %{{.*}}[symbol(%{{.*}})]
// CHECK: %[[INIT:.*]] = arith.addi
// CHECK: affine.for %[[I:.*]] = %{{.*}} to {{.*}} iter_args(%[[PREV:.*]] = %[[INIT]])
// CHECK: %[[CUR:.*]] = arith.addi
// CHECK: affine.yield %[[CUR]]
func.func @symbolic_lower_fixed_trip(%src0: memref<?xi32>,
                                    %dst0: memref<?xi32>, %start: index) {
  %src, %dst = memref.distinct_objects %src0, %dst0
      : memref<?xi32>, memref<?xi32>
  affine.for %i = #lb(%start) to #ub8(%start) {
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

// A statically empty symbolic interval must not cause a prologue load.

// CHECK-LABEL: func.func @symbolic_negative_trip
// CHECK-NOT: affine.load %{{.*}}[symbol(%{{.*}})]
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @symbolic_negative_trip(%src0: memref<?xi32>,
                                  %dst0: memref<?xi32>, %start: index) {
  %src, %dst = memref.distinct_objects %src0, %dst0
      : memref<?xi32>, memref<?xi32>
  affine.for %i = #lb(%start) to #ub_minus1(%start) {
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

// A signed-overflowing bound difference is still an empty interval.

// CHECK-LABEL: func.func @constant_overflow_empty
// CHECK-NOT: affine.load
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @constant_overflow_empty(%src0: memref<?xi32>,
                                   %dst0: memref<?xi32>) {
  %src, %dst = memref.distinct_objects %src0, %dst0
      : memref<?xi32>, memref<?xi32>
  affine.for %i = 9223372036854775807 to -9223372036854775800 {
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

// One executing iteration cannot eliminate a repeated producer evaluation.

// CHECK-LABEL: func.func @one_trip
// CHECK-NOT: affine.load %{{.*}}[0]
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @one_trip(%src0: memref<3xi32>, %dst0: memref<1xi32>) {
  %src, %dst = memref.distinct_objects %src0, %dst0
      : memref<3xi32>, memref<1xi32>
  affine.for %i = 0 to 1 {
    %a = affine.load %src[%i] : memref<3xi32>
    %b = affine.load %src[%i + 1] : memref<3xi32>
    %c = affine.load %src[%i + 2] : memref<3xi32>
    %left = arith.addi %a, %b : i32
    %right = arith.addi %b, %c : i32
    %difference = arith.subi %right, %left : i32
    affine.store %difference, %dst[%i] : memref<1xi32>
  }
  return
}

// CHECK-LABEL: func.func @zero_trip
// CHECK-NOT: affine.load
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @zero_trip(%src0: memref<2xi32>, %dst0: memref<1xi32>) {
  %src, %dst = memref.distinct_objects %src0, %dst0
      : memref<2xi32>, memref<1xi32>
  affine.for %i = 0 to 0 {
    %a = affine.load %src[%i] : memref<2xi32>
    %b = affine.load %src[%i + 1] : memref<2xi32>
    %left = arith.addi %a, %b : i32
    %right = arith.addi %b, %a : i32
    %difference = arith.subi %right, %left : i32
    affine.store %difference, %dst[%i] : memref<1xi32>
  }
  return
}

// Operation attributes are part of the producer semantics.

// CHECK-LABEL: func.func @attribute_mismatch
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @attribute_mismatch(%src0: memref<10xi32>, %dst0: memref<8xi32>) {
  %src, %dst = memref.distinct_objects %src0, %dst0
      : memref<10xi32>, memref<8xi32>
  affine.for %i = 0 to 8 {
    %a = affine.load %src[%i] : memref<10xi32>
    %b = affine.load %src[%i + 1] : memref<10xi32>
    %c = affine.load %src[%i + 2] : memref<10xi32>
    %left = arith.addi %a, %b overflow<nsw> : i32
    %right = arith.addi %b, %c : i32
    %difference = arith.subi %right, %left : i32
    affine.store %difference, %dst[%i] : memref<8xi32>
  }
  return
}

// Equal-looking computations from different memory objects are not a
// translated producer pair.

// CHECK-LABEL: func.func @different_sources
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @different_sources(%a0: memref<10xi32>, %b0: memref<10xi32>,
                             %dst0: memref<8xi32>) {
  %a, %b, %dst = memref.distinct_objects %a0, %b0, %dst0
      : memref<10xi32>, memref<10xi32>, memref<8xi32>
  affine.for %i = 0 to 8 {
    %a1 = affine.load %a[%i] : memref<10xi32>
    %a2 = affine.load %a[%i + 1] : memref<10xi32>
    %b1 = affine.load %b[%i + 1] : memref<10xi32>
    %b2 = affine.load %b[%i + 2] : memref<10xi32>
    %left = arith.addi %a1, %a2 : i32
    %right = arith.addi %b1, %b2 : i32
    %difference = arith.subi %right, %left : i32
    affine.store %difference, %dst[%i] : memref<8xi32>
  }
  return
}

// Repeated loop-invariant computations belong to LICM/CSE, not translated
// loop-carried reuse.

// CHECK-LABEL: func.func @loop_invariant_duplicates
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @loop_invariant_duplicates(%src0: memref<1xi32>,
                                     %dst0: memref<8xi32>) {
  %src, %dst = memref.distinct_objects %src0, %dst0
      : memref<1xi32>, memref<8xi32>
  affine.for %i = 0 to 8 {
    %a = affine.load %src[0] : memref<1xi32>
    %b = affine.load %src[0] : memref<1xi32>
    %left = arith.muli %a, %a : i32
    %right = arith.muli %b, %b : i32
    %difference = arith.subi %right, %left : i32
    affine.store %difference, %dst[%i] : memref<8xi32>
  }
  return
}

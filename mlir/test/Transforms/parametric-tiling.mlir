// RUN: mlir-opt -test-extract-fixed-outer-loops='test-outer-loop-sizes=7' %s | FileCheck %s --check-prefixes=COMMON,TILE_7
// RUN: mlir-opt -test-extract-fixed-outer-loops='test-outer-loop-sizes=7,4' %s | FileCheck %s --check-prefixes=COMMON,TILE_74

// COMMON-LABEL: @rectangular
func.func @rectangular(%arg0: memref<?x?xf32>) {
  %c2 = arith.constant 2 : index
  %c44 = arith.constant 44 : index
  %c1 = arith.constant 1 : index
  // Range of the original loop:
  //   (upper - lower + step - 1) / step
  // where step is known to be %c1.
  // COMMON:      %[[diff:.*]] = arith.subi %c44, %c2
  // COMMON:      %[[adjustment:.*]] = arith.subi %c1, %c1_{{.*}}
  // COMMON-NEXT: %[[diff_adj:.*]] = arith.addi %[[diff]], %[[adjustment]]
  // COMMON-NEXT: %[[range:.*]] = arith.divui %[[diff_adj]], %c1

  // Ceildiv to get the parametric tile size.
  // COMMON:       %[[sum:.*]] = arith.addi %[[range]], %c6
  // COMMON-NEXT:  %[[size:.*]] = arith.divui %[[sum]], %c7
  // New outer step (original is %c1).
  // COMMON-NEXT:      %[[step:.*]] = arith.muli %c1, %[[size]]

  // Range of the second original loop
  //   (upper - lower + step - 1) / step
  // where step is known to be %c2.
  // TILE_74:      %[[diff2:.*]] = arith.subi %c44, %c1
  // TILE_74:      %[[adjustment2:.*]] = arith.subi %c2, %c1_{{.*}}
  // TILE_74-NEXT: %[[diff2_adj:.*]] = arith.addi %[[diff2]], %[[adjustment2]]
  // TILE_74-NEXT: %[[range2:.*]] = arith.divui %[[diff2_adj]], %c2

  // Ceildiv to get the parametric tile size for the second original scf.
  // TILE_74:      %[[sum2:.*]] = arith.addi %[[range2]], %c3
  // TILE_74-NEXT: %[[size2:.*]] = arith.divui %[[sum2]], %c4
  // New inner step (original is %c2).
  // TILE_74-NEXT:     %[[step2:.*]] = arith.muli %c2, %[[size2]]

  // Updated outer loop(s) use new steps.
  // COMMON: scf.for %[[i:.*]] = %c2 to %c44 step %[[step]]
  // TILE_74:scf.for %[[j:.*]] = %c1 to %c44 step %[[step2]]
 scf.for %i = %c2 to %c44 step %c1 {
    // Upper bound for the inner loop min(%i + %step, %c44).
    // COMMON:      %[[stepped:.*]] = arith.addi %[[i]], %[[step]]
    // COMMON-NEXT: %[[ub:.*]] = arith.minsi %c44, %[[stepped]]
    //
    // TILE_74:      %[[stepped2:.*]] = arith.addi %[[j]], %[[step2]]
    // TILE_74-NEXT: %[[ub2:.*]] = arith.minsi %c44, %[[stepped2]]

    // Created inner scf.
    // COMMON:scf.for %[[ii:.*]] = %[[i]] to %[[ub:.*]] step %c1

    // This loop is not modified in TILE_7 case.
    // TILE_7: scf.for %[[j:.*]] = %c1 to %c44 step %c2
    //
    // But is modified in TILE_74 case.
    // TILE_74:scf.for %[[jj:.*]] = %[[j]] to %[[ub2]] step %c2
   scf.for %j = %c1 to %c44 step %c2 {
      // The right iterator are used.
      // TILE_7:  memref.load %arg0[%[[ii]], %[[j]]]
      // TILE_74: memref.load %arg0[%[[ii]], %[[jj]]]
      memref.load %arg0[%i, %j]: memref<?x?xf32>
    }
  }
  return
}

// COMMON-LABEL: @triangular
func.func @triangular(%arg0: memref<?x?xf32>) {
  %c2 = arith.constant 2 : index
  %c44 = arith.constant 44 : index
  %c1 = arith.constant 1 : index
  // Range of the original outer loop:
  //   (upper - lower + step - 1) / step
  // where step is known to be %c1.
  // COMMON:      %[[diff:.*]] = arith.subi %c44, %c2
  // COMMON:      %[[adjustment:.*]] = arith.subi %c1, %c1_{{.*}}
  // COMMON-NEXT: %[[diff_adj:.*]] = arith.addi %[[diff]], %[[adjustment]]
  // COMMON-NEXT: %[[range:.*]] = arith.divui %[[diff_adj]], %c1

  // Ceildiv to get the parametric tile size.
  // COMMON:       %[[sum:.*]] = arith.addi %[[range]], %c6
  // COMMON-NEXT:  %[[size:.*]] = arith.divui %[[sum]], %c7
  // New outer step (original is %c1).
  // COMMON-NEXT:  %[[step:.*]] = arith.muli %c1, %[[size]]

  // Constant adjustment for inner loop has been hoisted out.
  // TILE_74:      %[[adjustment2:.*]] = arith.subi %c2, %c1_{{.*}}

  // New outer scf.
  // COMMON: scf.for %[[i:.*]] = %c2 to %c44 step %[[step]]

  // Range of the original inner loop
  //   (upper - lower + step - 1) / step
  // where step is known to be %c2.
  // TILE_74:      %[[diff2:.*]] = arith.subi %[[i]], %c1
  // TILE_74-NEXT: %[[diff2_adj:.*]] = arith.addi %[[diff2]], %[[adjustment2]]
  // TILE_74-NEXT: %[[range2:.*]] = arith.divui %[[diff2_adj]], %c2

  // Ceildiv to get the parametric tile size for the second original scf.
  // TILE_74:      %[[sum2:.*]] = arith.addi %[[range2]], %c3
  // TILE_74-NEXT: %[[size2:.*]] = arith.divui %[[sum2]], %c4
  // New inner step (original is %c2).
  // TILE_74-NEXT:     %[[step2:.*]] = arith.muli %c2, %[[size2]]

  // New inner scf.
  // TILE_74:scf.for %[[j:.*]] = %c1 to %[[i]] step %[[step2]]
 scf.for %i = %c2 to %c44 step %c1 {
    // Upper bound for the inner loop min(%i + %step, %c44).
    // COMMON:      %[[stepped:.*]] = arith.addi %[[i]], %[[step]]
    // COMMON-NEXT: %[[ub:.*]] = arith.minsi %c44, %[[stepped]]
    // TILE_74:      %[[stepped2:.*]] = arith.addi %[[j]], %[[step2]]
    // TILE_74-NEXT: %[[ub2:.*]] = arith.minsi %[[i]], %[[stepped2]]
    //
    // Created inner scf.
    // COMMON:scf.for %[[ii:.*]] = %[[i]] to %[[ub:.*]] step %c1

    // This loop is not modified in TILE_7 case.
    // TILE_7: scf.for %[[j:.*]] = %c1 to %[[ii]] step %c2
    //
    // But is modified in TILE_74 case.
    // TILE_74:scf.for %[[jj:.*]] = %[[j]] to %[[ub2]] step %c2
   scf.for %j = %c1 to %i step %c2 {
      // The right iterator are used.
      // TILE_7:  memref.load %arg0[%[[ii]], %[[j]]]
      // TILE_74: memref.load %arg0[%[[ii]], %[[jj]]]
      memref.load %arg0[%i, %j]: memref<?x?xf32>
    }
  }
  return
}

// Verify that iter_args are threaded through the new inner loop.

// COMMON-LABEL: func.func @loop_with_iter_args(
// COMMON:         %[[r:.*]] = scf.for %[[i:.*]] = {{.*}} iter_args(%[[sum:.*]] = %{{.*}}) -> (f32) {
// COMMON:           %[[inner:.*]] = scf.for %[[ii:.*]] = %[[i]] to {{.*}} iter_args(%[[s:.*]] = %[[sum]]) -> (f32) {
// COMMON-NEXT:        %[[e:.*]] = memref.load %{{.*}}[%[[ii]]]
// COMMON-NEXT:        %[[new:.*]] = arith.addf %[[s]], %[[e]]
// COMMON-NEXT:        scf.yield %[[new]]
// COMMON-NEXT:      }
// COMMON-NEXT:      scf.yield %[[inner]]
// COMMON-NEXT:    }
// COMMON-NEXT:    return %[[r]]
func.func @loop_with_iter_args(%buffer: memref<1024xf32>, %lb: index,
                               %ub: index, %step: index) -> f32 {
  %initial_sum = arith.constant 0.0 : f32
  %final_sum = scf.for %iv = %lb to %ub step %step
      iter_args(%sum_iter = %initial_sum) -> (f32) {
    %element = memref.load %buffer[%iv] : memref<1024xf32>
    %updated_sum = arith.addf %sum_iter, %element : f32
    scf.yield %updated_sum : f32
  }
  return %final_sum : f32
}

// Verify that loops using unsigned comparisons are strip-mined with unsigned
// arithmetic: the upper bound of each strip is computed with `arith.minui` and
// the new inner loop also compares unsigned (regression test for
// https://github.com/llvm/llvm-project/issues/240306).

// COMMON-LABEL: func.func @loop_with_unsigned_cmp(
// COMMON-SAME:    %{{[^:]*}}: memref<1024xf32>, %[[lb:[^:]*]]: index, %[[ub:[^:]*]]: index, %[[step:[^:]*]]: index
// COMMON:         scf.for unsigned %[[i:.*]] = %[[lb]] to %[[ub]] step %[[tile:.*]] {
// COMMON-NEXT:      %[[stepped:.*]] = arith.addi %[[i]], %[[tile]]
// COMMON-NEXT:      %[[strip_ub:.*]] = arith.minui %[[ub]], %[[stepped]]
// COMMON-NEXT:      scf.for unsigned %[[ii:.*]] = %[[i]] to %[[strip_ub]] step %[[step]] {
// COMMON-NEXT:        memref.load %{{.*}}[%[[ii]]]
func.func @loop_with_unsigned_cmp(%buffer: memref<1024xf32>, %lb: index,
                                  %ub: index, %step: index) {
  scf.for unsigned %iv = %lb to %ub step %step {
    %element = memref.load %buffer[%iv] : memref<1024xf32>
    memref.store %element, %buffer[%iv] : memref<1024xf32>
  }
  return
}

// Verify that each new inner loop takes the signedness of the loop it was
// strip-mined from, not of the loop it is placed in. With two tiling levels,
// the strip of the unsigned outer loop is placed inside the signed loop.

// COMMON-LABEL: func.func @mixed_signedness(
// COMMON-SAME:    %{{[^:]*}}: memref<?x?xf32>, %[[n:[^:]*]]: index, %[[m:[^:]*]]: index
// COMMON:         scf.for unsigned %[[i:.*]] = %c0 to %[[n]] step %[[tile:.*]] {
// TILE_74-NEXT:     scf.for %[[j:.*]] = %c0 to %[[m]] step %[[tile2:.*]] {
// COMMON-NEXT:        %[[stepped:.*]] = arith.addi %[[i]], %[[tile]]
// COMMON-NEXT:        %[[strip_ub:.*]] = arith.minui %[[n]], %[[stepped]]
// TILE_74-NEXT:       %[[stepped2:.*]] = arith.addi %[[j]], %[[tile2]]
// TILE_74-NEXT:       %[[strip_ub2:.*]] = arith.minsi %[[m]], %[[stepped2]]
// COMMON-NEXT:        scf.for unsigned %[[ii:.*]] = %[[i]] to %[[strip_ub]] step %c1 {
// TILE_7-NEXT:          scf.for %[[j:.*]] = %c0 to %[[m]] step %c1 {
// TILE_74-NEXT:         scf.for %[[jj:.*]] = %[[j]] to %[[strip_ub2]] step %c1 {
// TILE_7-NEXT:            memref.load %{{.*}}[%[[ii]], %[[j]]]
// TILE_74-NEXT:           memref.load %{{.*}}[%[[ii]], %[[jj]]]
func.func @mixed_signedness(%arg0: memref<?x?xf32>, %n: index, %m: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for unsigned %i = %c0 to %n step %c1 {
    scf.for %j = %c0 to %m step %c1 {
      memref.load %arg0[%i, %j] : memref<?x?xf32>
    }
  }
  return
}

// Verify that a nest with iter_args is only tiled when a single loop is
// strip-mined, since tiling more loops reorders the iterations.

// COMMON-LABEL: func.func @nest_with_iter_args(
// TILE_7:         scf.for %[[i:.*]] = {{.*}} iter_args(%[[a:.*]] = %{{.*}}) -> (f32) {
// TILE_7:           scf.for %[[ii:.*]] = %[[i]] to {{.*}} iter_args(%[[a2:.*]] = %[[a]]) -> (f32) {
// TILE_7-NEXT:        scf.for %[[j:.*]] = {{.*}} iter_args(%[[b:.*]] = %[[a2]]) -> (f32) {
// TILE_7-NEXT:          memref.load %{{.*}}[%[[ii]], %[[j]]]
// TILE_74:        scf.for %[[i:.*]] = %c0 to %{{.*}} step %c1 iter_args
// TILE_74-NEXT:     scf.for %[[j:.*]] = %c0 to %{{.*}} step %c1 iter_args
// TILE_74-NEXT:       memref.load %{{.*}}[%[[i]], %[[j]]]
func.func @nest_with_iter_args(%buffer: memref<?x?xf32>, %n: index, %m: index,
                               %init: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%a = %init) -> (f32) {
    %r2 = scf.for %j = %c0 to %m step %c1 iter_args(%b = %a) -> (f32) {
      %e = memref.load %buffer[%i, %j] : memref<?x?xf32>
      %s = arith.addf %b, %e : f32
      scf.yield %s : f32
    }
    scf.yield %r2 : f32
  }
  return %r : f32
}

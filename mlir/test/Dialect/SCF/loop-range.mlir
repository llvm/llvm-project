// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(scf-for-loop-range-folding))' -split-input-file | FileCheck %s

func.func @fold_one_loop(%arg0: memref<?xi32>, %arg1: index, %arg2: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %arg1 step %c1 {
    %0 = arith.addi %arg2, %i : index
    %1 = arith.muli %0, %c4 : index
    %2 = memref.load %arg0[%1] : memref<?xi32>
    %3 = arith.muli %2, %2 : i32
    memref.store %3, %arg0[%1] : memref<?xi32>
  }
  return
}

// CHECK-LABEL: func @fold_one_loop
// CHECK-SAME:   (%[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %[[ARG2:.*]]: {{.*}}
// CHECK:       %[[C0:.*]] = arith.constant 0 : index
// CHECK:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:       %[[C4:.*]] = arith.constant 4 : index
// CHECK:       %[[I0:.*]] = arith.addi %[[ARG2]], %[[C0]] : index
// CHECK:       %[[I1:.*]] = arith.addi %[[ARG2]], %[[ARG1]] : index
// CHECK:       %[[I2:.*]] = arith.muli %[[I0]], %[[C4]] : index
// CHECK:       %[[I3:.*]] = arith.muli %[[I1]], %[[C4]] : index
// CHECK:       %[[I4:.*]] = arith.muli %[[C1]], %[[C4]] : index
// CHECK:       scf.for %[[I:.*]] = %[[I2]] to %[[I3]] step %[[I4]] {
// CHECK:         %[[I5:.*]] = memref.load %[[ARG0]]{{\[}}%[[I]]
// CHECK:         %[[I6:.*]] = arith.muli %[[I5]], %[[I5]] : i32
// CHECK:         memref.store %[[I6]], %[[ARG0]]{{\[}}%[[I]]

func.func @fold_one_loop2(%arg0: memref<?xi32>, %arg1: index, %arg2: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c10 = arith.constant 10 : index
  scf.for %j = %c0 to %c10 step %c1 {
    scf.for %i = %c0 to %arg1 step %c1 {
      %0 = arith.addi %arg2, %i : index
      %1 = arith.muli %0, %c4 : index
      %2 = memref.load %arg0[%1] : memref<?xi32>
      %3 = arith.muli %2, %2 : i32
      memref.store %3, %arg0[%1] : memref<?xi32>
    }
  }
  return
}

// CHECK-LABEL: func @fold_one_loop2
// CHECK-SAME:   (%[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %[[ARG2:.*]]: {{.*}}
// CHECK:       %[[C0:.*]] = arith.constant 0 : index
// CHECK:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:       %[[C4:.*]] = arith.constant 4 : index
// CHECK:       %[[C10:.*]] = arith.constant 10 : index
// CHECK:       scf.for %[[J:.*]] = %[[C0]] to %[[C10]] step %[[C1]] {
// CHECK:         %[[I0:.*]] = arith.addi %[[ARG2]], %[[C0]] : index
// CHECK:         %[[I1:.*]] = arith.addi %[[ARG2]], %[[ARG1]] : index
// CHECK:         %[[I2:.*]] = arith.muli %[[I0]], %[[C4]] : index
// CHECK:         %[[I3:.*]] = arith.muli %[[I1]], %[[C4]] : index
// CHECK:         %[[I4:.*]] = arith.muli %[[C1]], %[[C4]] : index
// CHECK:         scf.for %[[I:.*]] = %[[I2]] to %[[I3]] step %[[I4]] {
// CHECK:           %[[I5:.*]] = memref.load %[[ARG0]]{{\[}}%[[I]]
// CHECK:           %[[I6:.*]] = arith.muli %[[I5]], %[[I5]] : i32
// CHECK:           memref.store %[[I6]], %[[ARG0]]{{\[}}%[[I]]

func.func @fold_two_loops(%arg0: memref<?xi32>, %arg1: index, %arg2: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c10 = arith.constant 10 : index
  scf.for %j = %c0 to %c10 step %c1 {
    scf.for %i = %j to %arg1 step %c1 {
      %0 = arith.addi %arg2, %i : index
      %1 = arith.muli %0, %c4 : index
      %2 = memref.load %arg0[%1] : memref<?xi32>
      %3 = arith.muli %2, %2 : i32
      memref.store %3, %arg0[%1] : memref<?xi32>
    }
  }
  return
}

// CHECK-LABEL: func @fold_two_loops
// CHECK-SAME:   (%[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %[[ARG2:.*]]: {{.*}}
// CHECK:       %[[C0:.*]] = arith.constant 0 : index
// CHECK:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:       %[[C4:.*]] = arith.constant 4 : index
// CHECK:       %[[C10:.*]] = arith.constant 10 : index
// CHECK:       %[[I0:.*]] = arith.addi %[[ARG2]], %[[C0]] : index
// CHECK:       %[[I1:.*]] = arith.addi %[[ARG2]], %[[C10]] : index
// CHECK:       %[[I2:.*]] = arith.muli %[[I0]], %[[C4]] : index
// CHECK:       %[[I3:.*]] = arith.muli %[[I1]], %[[C4]] : index
// CHECK:       %[[I4:.*]] = arith.muli %[[C1]], %[[C4]] : index
// CHECK:       scf.for %[[J:.*]] = %[[I2]] to %[[I3]] step %[[I4]] {
// CHECK:         %[[I5:.*]] = arith.addi %[[ARG2]], %[[ARG1]] : index
// CHECK:         %[[I6:.*]] = arith.muli %[[I5]], %[[C4]] : index
// CHECK:         %[[I7:.*]] = arith.muli %[[C1]], %[[C4]] : index
// CHECK:         scf.for %[[I:.*]] = %[[J]] to %[[I6]] step %[[I7]] {
// CHECK:           %[[I8:.*]] = memref.load %[[ARG0]]{{\[}}%[[I]]
// CHECK:           %[[I9:.*]] = arith.muli %[[I8]], %[[I8]] : i32
// CHECK:           memref.store %[[I9]], %[[ARG0]]{{\[}}%[[I]]

// If an instruction's operands are not defined outside the loop, we cannot
// perform the optimization, as is the case with the arith.muli below. (If
// paired with loop invariant code motion we can continue.)
func.func @fold_only_first_add(%arg0: memref<?xi32>, %arg1: index, %arg2: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %arg1 step %c1 {
    %0 = arith.addi %arg2, %i : index
    %1 = arith.addi %arg2, %c4 : index
    %2 = arith.muli %0, %1 : index
    %3 = memref.load %arg0[%2] : memref<?xi32>
    %4 = arith.muli %3, %3 : i32
    memref.store %4, %arg0[%2] : memref<?xi32>
  }
  return
}

// CHECK-LABEL: func @fold_only_first_add
// CHECK-SAME:   (%[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %[[ARG2:.*]]: {{.*}}
// CHECK:       %[[C0:.*]] = arith.constant 0 : index
// CHECK:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:       %[[C4:.*]] = arith.constant 4 : index
// CHECK:       %[[I0:.*]] = arith.addi %[[ARG2]], %[[C0]] : index
// CHECK:       %[[I1:.*]] = arith.addi %[[ARG2]], %[[ARG1]] : index
// CHECK:       scf.for %[[I:.*]] = %[[I0]] to %[[I1]] step %[[C1]] {
// CHECK:         %[[I2:.*]] = arith.addi %[[ARG2]], %[[C4]] : index
// CHECK:         %[[I3:.*]] = arith.muli %[[I]], %[[I2]] : index
// CHECK:         %[[I4:.*]] = memref.load %[[ARG0]]{{\[}}%[[I3]]
// CHECK:         %[[I5:.*]] = arith.muli %[[I4]], %[[I4]] : i32
// CHECK:         memref.store %[[I5]], %[[ARG0]]{{\[}}%[[I3]]

// Do not fold arith.muli with a negative multiplier: the resulting loop step
// would be negative, which is invalid for scf.for.
func.func @no_fold_negative_mul() {
  %cm1 = arith.constant -1 : index
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c10 step %c1 {
    %0 = arith.muli %i, %cm1 : index
    "test.sink"(%0) : (index) -> ()
  }
  return
}

// CHECK-LABEL: func @no_fold_negative_mul
// CHECK:       %[[CM1:.*]] = arith.constant -1 : index
// CHECK:       %[[C0:.*]] = arith.constant 0 : index
// CHECK:       %[[C10:.*]] = arith.constant 10 : index
// CHECK:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:       scf.for %[[I:.*]] = %[[C0]] to %[[C10]] step %[[C1]] {
// CHECK:         %[[R:.*]] = arith.muli %[[I]], %[[CM1]] : index
// CHECK:         "test.sink"(%[[R]])

// Do not fold arith.muli with a negative multiplier when the induction variable
// is on the RHS.
func.func @no_fold_negative_mul_indvar_rhs() {
  %cm1 = arith.constant -1 : index
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c10 step %c1 {
    %0 = arith.muli %cm1, %i : index
    "test.sink"(%0) : (index) -> ()
  }
  return
}

// CHECK-LABEL: func @no_fold_negative_mul_indvar_rhs
// CHECK:       %[[CM1:.*]] = arith.constant -1 : index
// CHECK:       %[[C0:.*]] = arith.constant 0 : index
// CHECK:       %[[C10:.*]] = arith.constant 10 : index
// CHECK:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:       scf.for %[[I:.*]] = %[[C0]] to %[[C10]] step %[[C1]] {
// CHECK:         %[[R:.*]] = arith.muli %[[CM1]], %[[I]] : index
// CHECK:         "test.sink"(%[[R]])

// Do not fold arith.muli with a zero multiplier.
func.func @no_fold_zero_mul() {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c10 step %c1 {
    %0 = arith.muli %i, %c0 : index
    "test.sink"(%0) : (index) -> ()
  }
  return
}

// CHECK-LABEL: func @no_fold_zero_mul
// CHECK:       %[[C0:.*]] = arith.constant 0 : index
// CHECK:       %[[C10:.*]] = arith.constant 10 : index
// CHECK:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:       scf.for %[[I:.*]] = %[[C0]] to %[[C10]] step %[[C1]] {
// CHECK:         %[[R:.*]] = arith.muli %[[I]], %[[C0]] : index
// CHECK:         "test.sink"(%[[R]])

// Do not fold arith.muli when the multiplier is a non-constant loop-invariant
// value, since its sign is unknown at compile time.
func.func @no_fold_runtime_mul(%multiplier: index) {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c10 step %c1 {
    %0 = arith.muli %i, %multiplier : index
    "test.sink"(%0) : (index) -> ()
  }
  return
}

// CHECK-LABEL: func @no_fold_runtime_mul
// CHECK:       %[[C0:.*]] = arith.constant 0 : index
// CHECK:       %[[C10:.*]] = arith.constant 10 : index
// CHECK:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:       scf.for %[[I:.*]] = %[[C0]] to %[[C10]] step %[[C1]] {
// CHECK:         %[[R:.*]] = arith.muli %[[I]], {{.*}} : index
// CHECK:         "test.sink"(%[[R]])

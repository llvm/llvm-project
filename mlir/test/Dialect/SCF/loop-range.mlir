// RUN: mlir-opt %s -pass-pipeline='func(for-loop-range-folding)' -split-input-file | FileCheck %s

func @fold_one_loop(%arg0: memref<?xi32>, %arg1: index, %arg2: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c4 = constant 4 : index
  scf.for %i = %c0 to %arg1 step %c1 {
    %0 = addi %arg2, %i : index
    %1 = muli %0, %c4 : index
    %2 = memref.load %arg0[%1] : memref<?xi32>
    %3 = muli %2, %2 : i32
    memref.store %3, %arg0[%1] : memref<?xi32>
  }
  return
}

// CHECK-LABEL: func @fold_one_loop
// CHECK-SAME:   (%[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %[[ARG2:.*]]: {{.*}}
// CHECK:       %[[C0:.*]] = constant 0 : index
// CHECK:       %[[C1:.*]] = constant 1 : index
// CHECK:       %[[C4:.*]] = constant 4 : index
// CHECK:       %[[I0:.*]] = addi %[[ARG2]], %[[C0]] : index
// CHECK:       %[[I1:.*]] = addi %[[ARG2]], %[[ARG1]] : index
// CHECK:       %[[I2:.*]] = muli %[[I1]], %[[C4]] : index
// CHECK:       %[[I3:.*]] = muli %[[C1]], %[[C4]] : index
// CHECK:       scf.for %[[I:.*]] = %[[I0]] to %[[I2]] step %[[I3]] {
// CHECK:         %[[I4:.*]] = memref.load %[[ARG0]]{{\[}}%[[I]]
// CHECK:         %[[I5:.*]] = muli %[[I4]], %[[I4]] : i32
// CHECK:         memref.store %[[I5]], %[[ARG0]]{{\[}}%[[I]]

func @fold_one_loop2(%arg0: memref<?xi32>, %arg1: index, %arg2: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c4 = constant 4 : index
  %c10 = constant 10 : index
  scf.for %j = %c0 to %c10 step %c1 {
    scf.for %i = %c0 to %arg1 step %c1 {
      %0 = addi %arg2, %i : index
      %1 = muli %0, %c4 : index
      %2 = memref.load %arg0[%1] : memref<?xi32>
      %3 = muli %2, %2 : i32
      memref.store %3, %arg0[%1] : memref<?xi32>
    }
  }
  return
}

// CHECK-LABEL: func @fold_one_loop2
// CHECK-SAME:   (%[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %[[ARG2:.*]]: {{.*}}
// CHECK:       %[[C0:.*]] = constant 0 : index
// CHECK:       %[[C1:.*]] = constant 1 : index
// CHECK:       %[[C4:.*]] = constant 4 : index
// CHECK:       %[[C10:.*]] = constant 10 : index
// CHECK:       scf.for %[[J:.*]] = %[[C0]] to %[[C10]] step %[[C1]] {
// CHECK:         %[[I0:.*]] = addi %[[ARG2]], %[[C0]] : index
// CHECK:         %[[I1:.*]] = addi %[[ARG2]], %[[ARG1]] : index
// CHECK:         %[[I2:.*]] = muli %[[I1]], %[[C4]] : index
// CHECK:         %[[I3:.*]] = muli %[[C1]], %[[C4]] : index
// CHECK:         scf.for %[[I:.*]] = %[[I0]] to %[[I2]] step %[[I3]] {
// CHECK:           %[[I4:.*]] = memref.load %[[ARG0]]{{\[}}%[[I]]
// CHECK:           %[[I5:.*]] = muli %[[I4]], %[[I4]] : i32
// CHECK:           memref.store %[[I5]], %[[ARG0]]{{\[}}%[[I]]

func @fold_two_loops(%arg0: memref<?xi32>, %arg1: index, %arg2: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c4 = constant 4 : index
  %c10 = constant 10 : index
  scf.for %j = %c0 to %c10 step %c1 {
    scf.for %i = %j to %arg1 step %c1 {
      %0 = addi %arg2, %i : index
      %1 = muli %0, %c4 : index
      %2 = memref.load %arg0[%1] : memref<?xi32>
      %3 = muli %2, %2 : i32
      memref.store %3, %arg0[%1] : memref<?xi32>
    }
  }
  return
}

// CHECK-LABEL: func @fold_two_loops
// CHECK-SAME:   (%[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %[[ARG2:.*]]: {{.*}}
// CHECK:       %[[C0:.*]] = constant 0 : index
// CHECK:       %[[C1:.*]] = constant 1 : index
// CHECK:       %[[C4:.*]] = constant 4 : index
// CHECK:       %[[C10:.*]] = constant 10 : index
// CHECK:       %[[I0:.*]] = addi %[[ARG2]], %[[C0]] : index
// CHECK:       %[[I1:.*]] = addi %[[ARG2]], %[[C10]] : index
// CHECK:       scf.for %[[J:.*]] = %[[I0]] to %[[I1]] step %[[C1]] {
// CHECK:         %[[I1:.*]] = addi %[[ARG2]], %[[ARG1]] : index
// CHECK:         %[[I2:.*]] = muli %[[I1]], %[[C4]] : index
// CHECK:         %[[I3:.*]] = muli %[[C1]], %[[C4]] : index
// CHECK:         scf.for %[[I:.*]] = %[[J]] to %[[I2]] step %[[I3]] {
// CHECK:           %[[I4:.*]] = memref.load %[[ARG0]]{{\[}}%[[I]]
// CHECK:           %[[I5:.*]] = muli %[[I4]], %[[I4]] : i32
// CHECK:           memref.store %[[I5]], %[[ARG0]]{{\[}}%[[I]]

// If an instruction's operands are not defined outside the loop, we cannot
// perform the optimization, as is the case with the muli below. (If paired
// with loop invariant code motion we can continue.)
func @fold_only_first_add(%arg0: memref<?xi32>, %arg1: index, %arg2: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c4 = constant 4 : index
  scf.for %i = %c0 to %arg1 step %c1 {
    %0 = addi %arg2, %i : index
    %1 = addi %arg2, %c4 : index
    %2 = muli %0, %1 : index
    %3 = memref.load %arg0[%2] : memref<?xi32>
    %4 = muli %3, %3 : i32
    memref.store %4, %arg0[%2] : memref<?xi32>
  }
  return
}

// CHECK-LABEL: func @fold_only_first_add
// CHECK-SAME:   (%[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %[[ARG2:.*]]: {{.*}}
// CHECK:       %[[C0:.*]] = constant 0 : index
// CHECK:       %[[C1:.*]] = constant 1 : index
// CHECK:       %[[C4:.*]] = constant 4 : index
// CHECK:       %[[I0:.*]] = addi %[[ARG2]], %[[C0]] : index
// CHECK:       %[[I1:.*]] = addi %[[ARG2]], %[[ARG1]] : index
// CHECK:       scf.for %[[I:.*]] = %[[I0]] to %[[I1]] step %[[C1]] {
// CHECK:         %[[I2:.*]] = addi %[[ARG2]], %[[C4]] : index
// CHECK:         %[[I3:.*]] = muli %[[I]], %[[I2]] : index
// CHECK:         %[[I4:.*]] = memref.load %[[ARG0]]{{\[}}%[[I3]]
// CHECK:         %[[I5:.*]] = muli %[[I4]], %[[I4]] : i32
// CHECK:         memref.store %[[I5]], %[[ARG0]]{{\[}}%[[I3]]

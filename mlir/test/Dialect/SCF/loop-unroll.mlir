// RUN: mlir-opt %s -test-loop-unrolling='unroll-factor=2' | FileCheck %s --check-prefix UNROLL-BY-2
// RUN: mlir-opt %s -test-loop-unrolling='unroll-factor=3' | FileCheck %s --check-prefix UNROLL-BY-3
// RUN: mlir-opt %s -test-loop-unrolling='unroll-factor=2 loop-depth=0' | FileCheck %s --check-prefix UNROLL-OUTER-BY-2
// RUN: mlir-opt %s -test-loop-unrolling='unroll-factor=2 loop-depth=1' | FileCheck %s --check-prefix UNROLL-INNER-BY-2
// RUN: mlir-opt %s -test-loop-unrolling='unroll-factor=2 annotate=true' | FileCheck %s --check-prefix UNROLL-BY-2-ANNOTATE
// RUN: mlir-opt %s --affine-loop-unroll='unroll-factor=6 unroll-up-to-factor=true' | FileCheck %s --check-prefix UNROLL-UP-TO
// RUN: mlir-opt %s --affine-loop-unroll='unroll-factor=5 cleanup-unroll=true' | FileCheck %s --check-prefix CLEANUP-UNROLL-BY-5
// RUN: mlir-opt %s --affine-loop-unroll --split-input-file | FileCheck %s

func.func @dynamic_loop_unroll(%arg0 : index, %arg1 : index, %arg2 : index,
                          %arg3: memref<?xf32>) {
  %0 = arith.constant 7.0 : f32
  scf.for %i0 = %arg0 to %arg1 step %arg2 {
    memref.store %0, %arg3[%i0] : memref<?xf32>
  }
  return
}
// UNROLL-BY-2-LABEL: func @dynamic_loop_unroll
//  UNROLL-BY-2-SAME:  %[[LB:.*0]]: index,
//  UNROLL-BY-2-SAME:  %[[UB:.*1]]: index,
//  UNROLL-BY-2-SAME:  %[[STEP:.*2]]: index,
//  UNROLL-BY-2-SAME:  %[[MEM:.*3]]: memref<?xf32>
//
//   UNROLL-BY-2-DAG:  %[[V0:.*]] = arith.subi %[[UB]], %[[LB]] : index
//   UNROLL-BY-2-DAG:  %[[C1:.*]] = arith.constant 1 : index
//   UNROLL-BY-2-DAG:  %[[V1:.*]] = arith.subi %[[STEP]], %[[C1]] : index
//   UNROLL-BY-2-DAG:  %[[V2:.*]] = arith.addi %[[V0]], %[[V1]] : index
//       Compute trip count in V3.
//   UNROLL-BY-2-DAG:  %[[V3:.*]] = arith.divui %[[V2]], %[[STEP]] : index
//       Store unroll factor in C2.
//   UNROLL-BY-2-DAG:  %[[C2:.*]] = arith.constant 2 : index
//   UNROLL-BY-2-DAG:  %[[V4:.*]] = arith.remsi %[[V3]], %[[C2]] : index
//   UNROLL-BY-2-DAG:  %[[V5:.*]] = arith.subi %[[V3]], %[[V4]] : index
//   UNROLL-BY-2-DAG:  %[[V6:.*]] = arith.muli %[[V5]], %[[STEP]] : index
//       Compute upper bound of unrolled loop in V7.
//   UNROLL-BY-2-DAG:  %[[V7:.*]] = arith.addi %[[LB]], %[[V6]] : index
//       Compute step of unrolled loop in V8.
//   UNROLL-BY-2-DAG:  %[[V8:.*]] = arith.muli %[[STEP]], %[[C2]] : index
//       UNROLL-BY-2:  scf.for %[[IV:.*]] = %[[LB]] to %[[V7]] step %[[V8]] {
//  UNROLL-BY-2-NEXT:    memref.store %{{.*}}, %[[MEM]][%[[IV]]] : memref<?xf32>
//  UNROLL-BY-2-NEXT:    %[[C1_IV:.*]] = arith.constant 1 : index
//  UNROLL-BY-2-NEXT:    %[[V9:.*]] = arith.muli %[[STEP]], %[[C1_IV]] : index
//  UNROLL-BY-2-NEXT:    %[[V10:.*]] = arith.addi %[[IV]], %[[V9]] : index
//  UNROLL-BY-2-NEXT:    memref.store %{{.*}}, %[[MEM]][%[[V10]]] : memref<?xf32>
//  UNROLL-BY-2-NEXT:  }
//  UNROLL-BY-2-NEXT:  scf.for %[[IV:.*]] = %[[V7]] to %[[UB]] step %[[STEP]] {
//  UNROLL-BY-2-NEXT:    memref.store %{{.*}}, %[[MEM]][%[[IV]]] : memref<?xf32>
//  UNROLL-BY-2-NEXT:  }
//  UNROLL-BY-2-NEXT:  return

// UNROLL-BY-3-LABEL: func @dynamic_loop_unroll
//  UNROLL-BY-3-SAME:  %[[LB:.*0]]: index,
//  UNROLL-BY-3-SAME:  %[[UB:.*1]]: index,
//  UNROLL-BY-3-SAME:  %[[STEP:.*2]]: index,
//  UNROLL-BY-3-SAME:  %[[MEM:.*3]]: memref<?xf32>
//
//   UNROLL-BY-3-DAG:  %[[V0:.*]] = arith.subi %[[UB]], %[[LB]] : index
//   UNROLL-BY-3-DAG:  %[[C1:.*]] = arith.constant 1 : index
//   UNROLL-BY-3-DAG:  %[[V1:.*]] = arith.subi %[[STEP]], %[[C1]] : index
//   UNROLL-BY-3-DAG:  %[[V2:.*]] = arith.addi %[[V0]], %[[V1]] : index
//       Compute trip count in V3.
//   UNROLL-BY-3-DAG:  %[[V3:.*]] = arith.divui %[[V2]], %[[STEP]] : index
//       Store unroll factor in C3.
//   UNROLL-BY-3-DAG:  %[[C3:.*]] = arith.constant 3 : index
//   UNROLL-BY-3-DAG:  %[[V4:.*]] = arith.remsi %[[V3]], %[[C3]] : index
//   UNROLL-BY-3-DAG:  %[[V5:.*]] = arith.subi %[[V3]], %[[V4]] : index
//   UNROLL-BY-3-DAG:  %[[V6:.*]] = arith.muli %[[V5]], %[[STEP]] : index
//       Compute upper bound of unrolled loop in V7.
//   UNROLL-BY-3-DAG:  %[[V7:.*]] = arith.addi %[[LB]], %[[V6]] : index
//       Compute step of unrolled loop in V8.
//   UNROLL-BY-3-DAG:  %[[V8:.*]] = arith.muli %[[STEP]], %[[C3]] : index
//       UNROLL-BY-3:  scf.for %[[IV:.*]] = %[[LB]] to %[[V7]] step %[[V8]] {
//  UNROLL-BY-3-NEXT:    memref.store %{{.*}}, %[[MEM]][%[[IV]]] : memref<?xf32>
//  UNROLL-BY-3-NEXT:    %[[C1_IV:.*]] = arith.constant 1 : index
//  UNROLL-BY-3-NEXT:    %[[V9:.*]] = arith.muli %[[STEP]], %[[C1_IV]] : index
//  UNROLL-BY-3-NEXT:    %[[V10:.*]] = arith.addi %[[IV]], %[[V9]] : index
//  UNROLL-BY-3-NEXT:    memref.store %{{.*}}, %[[MEM]][%[[V10]]] : memref<?xf32>
//  UNROLL-BY-3-NEXT:    %[[C2_IV:.*]] = arith.constant 2 : index
//  UNROLL-BY-3-NEXT:    %[[V11:.*]] = arith.muli %[[STEP]], %[[C2_IV]] : index
//  UNROLL-BY-3-NEXT:    %[[V12:.*]] = arith.addi %[[IV]], %[[V11]] : index
//  UNROLL-BY-3-NEXT:    memref.store %{{.*}}, %[[MEM]][%[[V12]]] : memref<?xf32>
//  UNROLL-BY-3-NEXT:  }
//  UNROLL-BY-3-NEXT:  scf.for %[[IV:.*]] = %[[V7]] to %[[UB]] step %[[STEP]] {
//  UNROLL-BY-3-NEXT:    memref.store %{{.*}}, %[[MEM]][%[[IV]]] : memref<?xf32>
//  UNROLL-BY-3-NEXT:  }
//  UNROLL-BY-3-NEXT:  return

func.func @dynamic_loop_unroll_outer_by_2(
  %arg0 : index, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index,
  %arg5 : index, %arg6: memref<?xf32>) {
  %0 = arith.constant 7.0 : f32
  scf.for %i0 = %arg0 to %arg1 step %arg2 {
    scf.for %i1 = %arg3 to %arg4 step %arg5 {
     memref.store %0, %arg6[%i1] : memref<?xf32>
    }
  }
  return
}
// UNROLL-OUTER-BY-2-LABEL: func @dynamic_loop_unroll_outer_by_2
//  UNROLL-OUTER-BY-2-SAME:  %[[LB0:.*0]]: index,
//  UNROLL-OUTER-BY-2-SAME:  %[[UB0:.*1]]: index,
//  UNROLL-OUTER-BY-2-SAME:  %[[STEP0:.*2]]: index,
//  UNROLL-OUTER-BY-2-SAME:  %[[LB1:.*3]]: index,
//  UNROLL-OUTER-BY-2-SAME:  %[[UB1:.*4]]: index,
//  UNROLL-OUTER-BY-2-SAME:  %[[STEP1:.*5]]: index,
//  UNROLL-OUTER-BY-2-SAME:  %[[MEM:.*6]]: memref<?xf32>
//
//       UNROLL-OUTER-BY-2:  scf.for %[[IV0:.*]] = %[[LB0]] to %{{.*}} step %{{.*}} {
//  UNROLL-OUTER-BY-2-NEXT:    scf.for %[[IV1:.*]] = %[[LB1]] to %[[UB1]] step %[[STEP1]] {
//  UNROLL-OUTER-BY-2-NEXT:      memref.store %{{.*}}, %[[MEM]][%[[IV1]]] : memref<?xf32>
//  UNROLL-OUTER-BY-2-NEXT:    }
//  UNROLL-OUTER-BY-2-NEXT:    scf.for %[[IV1:.*]] = %[[LB1]] to %[[UB1]] step %[[STEP1]] {
//  UNROLL-OUTER-BY-2-NEXT:      memref.store %{{.*}}, %[[MEM]][%[[IV1]]] : memref<?xf32>
//  UNROLL-OUTER-BY-2-NEXT:    }
//  UNROLL-OUTER-BY-2-NEXT:  }
//  UNROLL-OUTER-BY-2-NEXT:  scf.for %[[IV0:.*]] = %{{.*}} to %[[UB0]] step %[[STEP0]] {
//  UNROLL-OUTER-BY-2-NEXT:    scf.for %[[IV1:.*]] = %[[LB1]] to %[[UB1]] step %[[STEP1]] {
//  UNROLL-OUTER-BY-2-NEXT:      memref.store %{{.*}}, %[[MEM]][%[[IV1]]] : memref<?xf32>
//  UNROLL-OUTER-BY-2-NEXT:    }
//  UNROLL-OUTER-BY-2-NEXT:  }
//  UNROLL-OUTER-BY-2-NEXT:  return

func.func @dynamic_loop_unroll_inner_by_2(
  %arg0 : index, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index,
  %arg5 : index, %arg6: memref<?xf32>) {
  %0 = arith.constant 7.0 : f32
  scf.for %i0 = %arg0 to %arg1 step %arg2 {
    scf.for %i1 = %arg3 to %arg4 step %arg5 {
     memref.store %0, %arg6[%i1] : memref<?xf32>
    }
  }
  return
}
// UNROLL-INNER-BY-2-LABEL: func @dynamic_loop_unroll_inner_by_2
//  UNROLL-INNER-BY-2-SAME:  %[[LB0:.*0]]: index,
//  UNROLL-INNER-BY-2-SAME:  %[[UB0:.*1]]: index,
//  UNROLL-INNER-BY-2-SAME:  %[[STEP0:.*2]]: index,
//  UNROLL-INNER-BY-2-SAME:  %[[LB1:.*3]]: index,
//  UNROLL-INNER-BY-2-SAME:  %[[UB1:.*4]]: index,
//  UNROLL-INNER-BY-2-SAME:  %[[STEP1:.*5]]: index,
//  UNROLL-INNER-BY-2-SAME:  %[[MEM:.*6]]: memref<?xf32>
//
//       UNROLL-INNER-BY-2:  scf.for %[[IV0:.*]] = %[[LB0]] to %[[UB0]] step %[[STEP0]] {
//       UNROLL-INNER-BY-2:    scf.for %[[IV1:.*]] = %[[LB1]] to %{{.*}} step %{{.*}} {
//  UNROLL-INNER-BY-2-NEXT:      memref.store %{{.*}}, %[[MEM]][%[[IV1]]] : memref<?xf32>
//  UNROLL-INNER-BY-2-NEXT:      %[[C1_IV:.*]] = arith.constant 1 : index
//  UNROLL-INNER-BY-2-NEXT:      %[[V0:.*]] = arith.muli %[[STEP1]], %[[C1_IV]] : index
//  UNROLL-INNER-BY-2-NEXT:      %[[V1:.*]] = arith.addi %[[IV1]], %[[V0]] : index
//  UNROLL-INNER-BY-2-NEXT:      memref.store %{{.*}}, %[[MEM]][%[[V1]]] : memref<?xf32>
//  UNROLL-INNER-BY-2-NEXT:    }
//  UNROLL-INNER-BY-2-NEXT:    scf.for %[[IV1:.*]] = %{{.*}} to %[[UB1]] step %[[STEP1]] {
//  UNROLL-INNER-BY-2-NEXT:      memref.store %{{.*}}, %[[MEM]][%[[IV1]]] : memref<?xf32>
//  UNROLL-INNER-BY-2-NEXT:    }
//  UNROLL-INNER-BY-2-NEXT:  }
//  UNROLL-INNER-BY-2-NEXT:  return

// Test that no epilogue clean-up loop is generated because the trip count is
// a multiple of the unroll factor.
func.func @static_loop_unroll_by_2(%arg0 : memref<?xf32>) {
  %0 = arith.constant 7.0 : f32
  %lb = arith.constant 0 : index
  %ub = arith.constant 20 : index
  %step = arith.constant 1 : index
  scf.for %i0 = %lb to %ub step %step {
    memref.store %0, %arg0[%i0] : memref<?xf32>
  }
  return
}
// UNROLL-BY-2-LABEL: func @static_loop_unroll_by_2
//  UNROLL-BY-2-SAME:  %[[MEM:.*0]]: memref<?xf32>
//
//   UNROLL-BY-2-DAG:  %[[C0:.*]] = arith.constant 0 : index
//   UNROLL-BY-2-DAG:  %[[C1:.*]] = arith.constant 1 : index
//   UNROLL-BY-2-DAG:  %[[C20:.*]] = arith.constant 20 : index
//   UNROLL-BY-2-DAG:  %[[C2:.*]] = arith.constant 2 : index
//   UNROLL-BY-2:  scf.for %[[IV:.*]] = %[[C0]] to %[[C20]] step %[[C2]] {
//  UNROLL-BY-2-NEXT:    memref.store %{{.*}}, %[[MEM]][%[[IV]]] : memref<?xf32>
//  UNROLL-BY-2-NEXT:    %[[C1_IV:.*]] = arith.constant 1 : index
//  UNROLL-BY-2-NEXT:    %[[V0:.*]] = arith.muli %[[C1]], %[[C1_IV]] : index
//  UNROLL-BY-2-NEXT:    %[[V1:.*]] = arith.addi %[[IV]], %[[V0]] : index
//  UNROLL-BY-2-NEXT:    memref.store %{{.*}}, %[[MEM]][%[[V1]]] : memref<?xf32>
//  UNROLL-BY-2-NEXT:  }
//  UNROLL-BY-2-NEXT:  return

// UNROLL-BY-2-ANNOTATE-LABEL: func @static_loop_unroll_by_2
// UNROLL-BY-2-ANNOTATE:    memref.store %{{.*}}, %[[MEM:.*0]][%{{.*}}] {unrolled_iteration = 0 : ui32} : memref<?xf32>
// UNROLL-BY-2-ANNOTATE:    memref.store %{{.*}}, %[[MEM]][%{{.*}}] {unrolled_iteration = 1 : ui32} : memref<?xf32>

// Test that no epilogue clean-up loop is generated because the trip count
// (taking into account the non-unit step size) is a multiple of the unroll
// factor.
func.func @static_loop_step_2_unroll_by_2(%arg0 : memref<?xf32>) {
  %0 = arith.constant 7.0 : f32
  %lb = arith.constant 0 : index
  %ub = arith.constant 19 : index
  %step = arith.constant 2 : index
  scf.for %i0 = %lb to %ub step %step {
    memref.store %0, %arg0[%i0] : memref<?xf32>
  }
  return
}

// UNROLL-BY-2-LABEL: func @static_loop_step_2_unroll_by_2
//  UNROLL-BY-2-SAME:  %[[MEM:.*0]]: memref<?xf32>
//
//   UNROLL-BY-2-DAG:  %[[C0:.*]] = arith.constant 0 : index
//   UNROLL-BY-2-DAG:  %[[C2:.*]] = arith.constant 2 : index
//   UNROLL-BY-2-DAG:  %[[C19:.*]] = arith.constant 19 : index
//   UNROLL-BY-2-DAG:  %[[C4:.*]] = arith.constant 4 : index
//   UNROLL-BY-2:  scf.for %[[IV:.*]] = %[[C0]] to %[[C19]] step %[[C4]] {
//  UNROLL-BY-2-NEXT:    memref.store %{{.*}}, %[[MEM]][%[[IV]]] : memref<?xf32>
//  UNROLL-BY-2-NEXT:    %[[C1_IV:.*]] = arith.constant 1 : index
//  UNROLL-BY-2-NEXT:    %[[V0:.*]] = arith.muli %[[C2]], %[[C1_IV]] : index
//  UNROLL-BY-2-NEXT:    %[[V1:.*]] = arith.addi %[[IV]], %[[V0]] : index
//  UNROLL-BY-2-NEXT:    memref.store %{{.*}}, %[[MEM]][%[[V1]]] : memref<?xf32>
//  UNROLL-BY-2-NEXT:  }
//  UNROLL-BY-2-NEXT:  return

// Test that epilogue clean up loop is generated (trip count is not
// a multiple of unroll factor).
func.func @static_loop_unroll_by_3(%arg0 : memref<?xf32>) {
  %0 = arith.constant 7.0 : f32
  %lb = arith.constant 0 : index
  %ub = arith.constant 20 : index
  %step = arith.constant 1 : index
  scf.for %i0 = %lb to %ub step %step {
    memref.store %0, %arg0[%i0] : memref<?xf32>
  }
  return
}

// UNROLL-BY-3-LABEL: func @static_loop_unroll_by_3
//  UNROLL-BY-3-SAME:  %[[MEM:.*0]]: memref<?xf32>
//
//   UNROLL-BY-3-DAG:  %[[C0:.*]] = arith.constant 0 : index
//   UNROLL-BY-3-DAG:  %[[C1:.*]] = arith.constant 1 : index
//   UNROLL-BY-3-DAG:  %[[C20:.*]] = arith.constant 20 : index
//   UNROLL-BY-3-DAG:  %[[C18:.*]] = arith.constant 18 : index
//   UNROLL-BY-3-DAG:  %[[C3:.*]] = arith.constant 3 : index
//       UNROLL-BY-3: scf.for %[[IV:.*]] = %[[C0]] to %[[C18]] step %[[C3]] {
//  UNROLL-BY-3-NEXT:    memref.store %{{.*}}, %[[MEM]][%[[IV]]] : memref<?xf32>
//  UNROLL-BY-3-NEXT:    %[[C1_IV:.*]] = arith.constant 1 : index
//  UNROLL-BY-3-NEXT:    %[[V0:.*]] = arith.muli %[[C1]], %[[C1_IV]] : index
//  UNROLL-BY-3-NEXT:    %[[V1:.*]] = arith.addi %[[IV]], %[[V0]] : index
//  UNROLL-BY-3-NEXT:    memref.store %{{.*}}, %[[MEM]][%[[V1]]] : memref<?xf32>
//  UNROLL-BY-3-NEXT:    %[[C2_IV:.*]] = arith.constant 2 : index
//  UNROLL-BY-3-NEXT:    %[[V2:.*]] = arith.muli %[[C1]], %[[C2_IV]] : index
//  UNROLL-BY-3-NEXT:    %[[V3:.*]] = arith.addi %[[IV]], %[[V2]] : index
//  UNROLL-BY-3-NEXT:    memref.store %{{.*}}, %[[MEM]][%[[V3]]] : memref<?xf32>
//  UNROLL-BY-3-NEXT:  }
//  UNROLL-BY-3-NEXT:  scf.for %[[IV:.*]] = %[[C18]] to %[[C20]] step %[[C1]] {
//  UNROLL-BY-3-NEXT:    memref.store %{{.*}}, %[[MEM]][%[[IV]]] : memref<?xf32>
//  UNROLL-BY-3-NEXT:  }
//  UNROLL-BY-3-NEXT:  return

// Test that the single iteration epilogue loop body is promoted to the loops
// containing block.
func.func @static_loop_unroll_by_3_promote_epilogue(%arg0 : memref<?xf32>) {
  %0 = arith.constant 7.0 : f32
  %lb = arith.constant 0 : index
  %ub = arith.constant 10 : index
  %step = arith.constant 1 : index
  scf.for %i0 = %lb to %ub step %step {
    memref.store %0, %arg0[%i0] : memref<?xf32>
  }
  return
}
// UNROLL-BY-3-LABEL: func @static_loop_unroll_by_3_promote_epilogue
//  UNROLL-BY-3-SAME:  %[[MEM:.*0]]: memref<?xf32>
//
//   UNROLL-BY-3-DAG:  %[[C0:.*]] = arith.constant 0 : index
//   UNROLL-BY-3-DAG:  %[[C1:.*]] = arith.constant 1 : index
//   UNROLL-BY-3-DAG:  %[[C10:.*]] = arith.constant 10 : index
//   UNROLL-BY-3-DAG:  %[[C9:.*]] = arith.constant 9 : index
//   UNROLL-BY-3-DAG:  %[[C3:.*]] = arith.constant 3 : index
//       UNROLL-BY-3: scf.for %[[IV:.*]] = %[[C0]] to %[[C9]] step %[[C3]] {
//  UNROLL-BY-3-NEXT:    memref.store %{{.*}}, %[[MEM]][%[[IV]]] : memref<?xf32>
//  UNROLL-BY-3-NEXT:    %[[C1_IV:.*]] = arith.constant 1 : index
//  UNROLL-BY-3-NEXT:    %[[V0:.*]] = arith.muli %[[C1]], %[[C1_IV]] : index
//  UNROLL-BY-3-NEXT:    %[[V1:.*]] = arith.addi %[[IV]], %[[V0]] : index
//  UNROLL-BY-3-NEXT:    memref.store %{{.*}}, %[[MEM]][%[[V1]]] : memref<?xf32>
//  UNROLL-BY-3-NEXT:    %[[C2_IV:.*]] = arith.constant 2 : index
//  UNROLL-BY-3-NEXT:    %[[V2:.*]] = arith.muli %[[C1]], %[[C2_IV]] : index
//  UNROLL-BY-3-NEXT:    %[[V3:.*]] = arith.addi %[[IV]], %[[V2]] : index
//  UNROLL-BY-3-NEXT:    memref.store %{{.*}}, %[[MEM]][%[[V3]]] : memref<?xf32>
//  UNROLL-BY-3-NEXT:  }
//  UNROLL-BY-3-NEXT:  memref.store %{{.*}}, %[[MEM]][%[[C9]]] : memref<?xf32>
//  UNROLL-BY-3-NEXT:  return

// Test unroll-up-to functionality.
func.func @static_loop_unroll_up_to_factor(%arg0 : memref<?xf32>) {
  %0 = arith.constant 7.0 : f32
  %lb = arith.constant 0 : index
  %ub = arith.constant 2 : index
  affine.for %i0 = %lb to %ub {
    affine.store %0, %arg0[%i0] : memref<?xf32>
  }
  return
}
// UNROLL-UP-TO-LABEL: func @static_loop_unroll_up_to_factor
//  UNROLL-UP-TO-SAME:  %[[MEM:.*0]]: memref<?xf32>
//
//   UNROLL-UP-TO-DAG:  %[[C0:.*]] = arith.constant 0 : index
//   UNROLL-UP-TO-DAG:  %[[C2:.*]] = arith.constant 2 : index
//   UNROLL-UP-TO-NEXT: %[[V0:.*]] = affine.apply {{.*}}
//   UNROLL-UP-TO-NEXT: affine.store %{{.*}}, %[[MEM]][%[[V0]]] : memref<?xf32>
//   UNROLL-UP-TO-NEXT: %[[V1:.*]] = affine.apply {{.*}}
//   UNROLL-UP-TO-NEXT: affine.store %{{.*}}, %[[MEM]][%[[V1]]] : memref<?xf32>
//   UNROLL-UP-TO-NEXT: return

// Test that epilogue's arguments are correctly renamed.
func.func @static_loop_unroll_by_3_rename_epilogue_arguments() -> (f32, f32) {
  %0 = arith.constant 7.0 : f32
  %lb = arith.constant 0 : index
  %ub = arith.constant 20 : index
  %step = arith.constant 1 : index
  %result:2 = scf.for %i0 = %lb to %ub step %step iter_args(%arg0 = %0, %arg1 = %0) -> (f32, f32) {
    %add = arith.addf %arg0, %arg1 : f32
    %mul = arith.mulf %arg0, %arg1 : f32
    scf.yield %add, %mul : f32, f32
  }
  return %result#0, %result#1 : f32, f32
}
// UNROLL-BY-3-LABEL: func @static_loop_unroll_by_3_rename_epilogue_arguments
//
//   UNROLL-BY-3-DAG:   %[[CST:.*]] = arith.constant {{.*}} : f32
//   UNROLL-BY-3-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   UNROLL-BY-3-DAG:   %[[C1:.*]] = arith.constant 1 : index
//   UNROLL-BY-3-DAG:   %[[C20:.*]] = arith.constant 20 : index
//   UNROLL-BY-3-DAG:   %[[C18:.*]] = arith.constant 18 : index
//   UNROLL-BY-3-DAG:   %[[C3:.*]] = arith.constant 3 : index
//       UNROLL-BY-3:   %[[FOR:.*]]:2 = scf.for %[[IV:.*]] = %[[C0]] to %[[C18]] step %[[C3]]
//  UNROLL-BY-3-SAME:     iter_args(%[[ARG0:.*]] = %[[CST]], %[[ARG1:.*]] = %[[CST]]) -> (f32, f32) {
//  UNROLL-BY-3-NEXT:     %[[ADD0:.*]] = arith.addf %[[ARG0]], %[[ARG1]] : f32
//  UNROLL-BY-3-NEXT:     %[[MUL0:.*]] = arith.mulf %[[ARG0]], %[[ARG1]] : f32
//  UNROLL-BY-3-NEXT:     %[[ADD1:.*]] = arith.addf %[[ADD0]], %[[MUL0]] : f32
//  UNROLL-BY-3-NEXT:     %[[MUL1:.*]] = arith.mulf %[[ADD0]], %[[MUL0]] : f32
//  UNROLL-BY-3-NEXT:     %[[ADD2:.*]] = arith.addf %[[ADD1]], %[[MUL1]] : f32
//  UNROLL-BY-3-NEXT:     %[[MUL2:.*]] = arith.mulf %[[ADD1]], %[[MUL1]] : f32
//  UNROLL-BY-3-NEXT:     scf.yield %[[ADD2]], %[[MUL2]] : f32, f32
//  UNROLL-BY-3-NEXT:   }
//       UNROLL-BY-3:   %[[EFOR:.*]]:2 = scf.for %[[EIV:.*]] = %[[C18]] to %[[C20]] step %[[C1]]
//  UNROLL-BY-3-SAME:     iter_args(%[[EARG0:.*]] = %[[FOR]]#0, %[[EARG1:.*]] = %[[FOR]]#1) -> (f32, f32) {
//  UNROLL-BY-3-NEXT:     %[[EADD:.*]] = arith.addf %[[EARG0]], %[[EARG1]] : f32
//  UNROLL-BY-3-NEXT:     %[[EMUL:.*]] = arith.mulf %[[EARG0]], %[[EARG1]] : f32
//  UNROLL-BY-3-NEXT:     scf.yield %[[EADD]], %[[EMUL]] : f32, f32
//  UNROLL-BY-3-NEXT:   }
//  UNROLL-BY-3-NEXT:   return %[[EFOR]]#0, %[[EFOR]]#1 : f32, f32

// Test that epilogue clean up loop is generated (trip count is less
// than an unroll factor).
func.func @static_loop_unroll_by_5_with_cleanup(%arg0 : memref<?xf32>) {
  %0 = arith.constant 7.0 : f32
  %lb = arith.constant 0 : index
  %ub = arith.constant 3 : index
  affine.for %i0 = %lb to %ub {
    memref.store %0, %arg0[%i0] : memref<?xf32>
  }
  return
}

// CLEANUP-UNROLL-BY-5-LABEL: func @static_loop_unroll_by_5_with_cleanup
//  CLEANUP-UNROLL-BY-5-SAME:  %[[MEM:.*0]]: memref<?xf32>
//
//   CLEANUP-UNROLL-BY-5-DAG:  %[[C0:.*]] = arith.constant 0 : index
//   CLEANUP-UNROLL-BY-5-DAG:  %[[C3:.*]] = arith.constant 3 : index
//   CLEANUP-UNROLL-BY-5-NEXT: %[[V0:.*]] = affine.apply {{.*}}
//   CLEANUP-UNROLL-BY-5-NEXT: memref.store %{{.*}}, %[[MEM]][%[[V0]]] : memref<?xf32>
//   CLEANUP-UNROLL-BY-5-NEXT: %[[V1:.*]] = affine.apply {{.*}}
//   CLEANUP-UNROLL-BY-5-NEXT: memref.store %{{.*}}, %[[MEM]][%[[V1]]] : memref<?xf32>
//   CLEANUP-UNROLL-BY-5-NEXT: %[[V2:.*]] = affine.apply {{.*}}
//   CLEANUP-UNROLL-BY-5-NEXT: memref.store %{{.*}}, %[[MEM]][%[[V2]]] : memref<?xf32>
//   CLEANUP-UNROLL-BY-5-NEXT: return

// -----

// Test loop unrolling when the yielded value remains unchanged.
// CHECK: [[$MAP:#map]] = affine_map<(d0) -> (-d0 + 64, (d0 floordiv 8) ceildiv 64, -d0 - 16, d0 * -64)>
// CHECK-LABEL: func @loop_unroll_static_yield_value
func.func @loop_unroll_static_yield_value_test1() {
  %true_4 = arith.constant true
  %c1 = arith.constant 1 : index
  %103 = affine.for %arg2 = 0 to 40 iter_args(%arg3 = %true_4) -> (i1) {
    %324 = affine.max affine_map<(d0) -> (-d0 + 64, (d0 floordiv 8) ceildiv 64, -d0 - 16, d0 * -64)>(%c1)
    affine.yield %true_4 : i1
  }
  return
}
// CHECK:         %[[TRUE:.*]] = arith.constant true
// CHECK-NEXT:    %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:    affine.for %{{.*}} = 0 to 40 step 4 iter_args(%{{.*}} = %[[TRUE]]) -> (i1) {
// CHECK-NEXT:      affine.max [[$MAP]](%[[C1]])
// CHECK-NEXT:      affine.max [[$MAP]](%[[C1]])
// CHECK-NEXT:      affine.max [[$MAP]](%[[C1]])
// CHECK-NEXT:      affine.max [[$MAP]](%[[C1]])
// CHECK-NEXT:      affine.yield %[[TRUE]] : i1
// CHECK-NEXT:    }
// CHECK-NEXT:    return

// -----

// Loop unrolling when the yielded value is loop iv.
// CHECK: [[$MAP0:#map[0-9]*]] = affine_map<(d0) -> (-d0 + 64, (d0 floordiv 8) ceildiv 64, -d0 - 16, d0 * -64)>
// CHECK: [[$MAP1:#map[0-9]*]] = affine_map<(d0) -> (d0 + 2)>
// CHECK: [[$MAP2:#map[0-9]*]] = affine_map<(d0) -> (d0 + 4)>
// CHECK: [[$MAP3:#map[0-9]*]] = affine_map<(d0) -> (d0 + 6)>
// CHECK-LABEL: func @loop_unroll_yield_loop_iv
func.func @loop_unroll_yield_loop_iv() {
  %c1 = arith.constant 1 : index
  %103 = affine.for %arg2 = 0 to 40 step 2 iter_args(%arg3 = %c1) -> (index) {
    %324 = affine.max affine_map<(d0) -> (-d0 + 64, (d0 floordiv 8) ceildiv 64, -d0 - 16, d0 * -64)>(%arg2)
    affine.yield %arg2 : index
  }
  return
}
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:    affine.for %[[LOOP_IV:.*]] = 0 to 40 step 8 iter_args(%{{.*}} = %[[C1]]) -> (index) {
// CHECK-NEXT:    affine.max [[$MAP0]](%[[LOOP_IV]])
// CHECK-NEXT:    %[[LOOP_IV_PLUS_2:.*]] = affine.apply [[$MAP1]](%[[LOOP_IV]])
// CHECK-NEXT:    affine.max [[$MAP0]](%[[LOOP_IV_PLUS_2]])
// CHECK-NEXT:    %[[LOOP_IV_PLUS_4:.*]] = affine.apply [[$MAP2]](%[[LOOP_IV]])
// CHECK-NEXT:    affine.max [[$MAP0]](%[[LOOP_IV_PLUS_4]])
// CHECK-NEXT:    %[[LOOP_IV_PLUS_6:.*]] = affine.apply [[$MAP3]](%[[LOOP_IV]])
// CHECK-NEXT:    affine.max [[$MAP0]](%[[LOOP_IV_PLUS_6]])
// CHECK-NEXT:    affine.yield %[[LOOP_IV]] : index
// CHECK-NEXT:  }
// CHECK-NEXT:  return

// -----

// Loop unrolling when the yielded value is iter_arg.
// CHECK: [[$MAP:#map]] = affine_map<(d0) -> (-d0 + 64, (d0 floordiv 8) ceildiv 64, -d0 - 16, d0 * -64)>
// CHECK-LABEL: func @loop_unroll_yield_iter_arg
func.func @loop_unroll_yield_iter_arg() {
  %c1 = arith.constant 1 : index
  %103 = affine.for %arg2 = 0 to 40 step 2 iter_args(%arg3 = %c1) -> (index) {
    %324 = affine.max affine_map<(d0) -> (-d0 + 64, (d0 floordiv 8) ceildiv 64, -d0 - 16, d0 * -64)>(%arg3)
    affine.yield %arg3 : index
  }
  return
}
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:    affine.for %{{.*}} = 0 to 40 step 8 iter_args(%[[ITER_ARG:.*]] = %[[C1]]) -> (index) {
// CHECK-NEXT:      affine.max [[$MAP]](%[[ITER_ARG]])
// CHECK-NEXT:      affine.max [[$MAP]](%[[ITER_ARG]])
// CHECK-NEXT:      affine.max [[$MAP]](%[[ITER_ARG]])
// CHECK-NEXT:      affine.max [[$MAP]](%[[ITER_ARG]])
// CHECK-NEXT:      affine.yield %[[ITER_ARG]] : index
// CHECK-NEXT:    }
// CHECK-NEXT:    return

// -----

// Test the loop unroller works with integer IV type.
func.func @static_loop_unroll_with_integer_iv() -> (f32, f32) {
  %0 = arith.constant 7.0 : f32
  %lb = arith.constant 0 : i32
  %ub = arith.constant 20 : i32
  %step = arith.constant 1 : i32
  %result:2 = scf.for %i0 = %lb to %ub step %step iter_args(%arg0 = %0, %arg1 = %0) -> (f32, f32) : i32{
    %add = arith.addf %arg0, %arg1 : f32
    %mul = arith.mulf %arg0, %arg1 : f32
    scf.yield %add, %mul : f32, f32
  }
  return %result#0, %result#1 : f32, f32
}
// UNROLL-BY-3-LABEL: func @static_loop_unroll_with_integer_iv
//
//   UNROLL-BY-3-DAG:   %[[CST:.*]] = arith.constant {{.*}} : f32
//   UNROLL-BY-3-DAG:   %[[C0:.*]] = arith.constant 0 : i32
//   UNROLL-BY-3-DAG:   %[[C1:.*]] = arith.constant 1 : i32
//   UNROLL-BY-3-DAG:   %[[C20:.*]] = arith.constant 20 : i32
//   UNROLL-BY-3-DAG:   %[[C18:.*]] = arith.constant 18 : i32
//   UNROLL-BY-3-DAG:   %[[C3:.*]] = arith.constant 3 : i32
//       UNROLL-BY-3:   %[[FOR:.*]]:2 = scf.for %[[IV:.*]] = %[[C0]] to %[[C18]] step %[[C3]]
//  UNROLL-BY-3-SAME:     iter_args(%[[ARG0:.*]] = %[[CST]], %[[ARG1:.*]] = %[[CST]]) -> (f32, f32)  : i32 {
//  UNROLL-BY-3-NEXT:     %[[ADD0:.*]] = arith.addf %[[ARG0]], %[[ARG1]] : f32
//  UNROLL-BY-3-NEXT:     %[[MUL0:.*]] = arith.mulf %[[ARG0]], %[[ARG1]] : f32
//  UNROLL-BY-3-NEXT:     %[[ADD1:.*]] = arith.addf %[[ADD0]], %[[MUL0]] : f32
//  UNROLL-BY-3-NEXT:     %[[MUL1:.*]] = arith.mulf %[[ADD0]], %[[MUL0]] : f32
//  UNROLL-BY-3-NEXT:     %[[ADD2:.*]] = arith.addf %[[ADD1]], %[[MUL1]] : f32
//  UNROLL-BY-3-NEXT:     %[[MUL2:.*]] = arith.mulf %[[ADD1]], %[[MUL1]] : f32
//  UNROLL-BY-3-NEXT:     scf.yield %[[ADD2]], %[[MUL2]] : f32, f32
//  UNROLL-BY-3-NEXT:   }
//       UNROLL-BY-3:   %[[EFOR:.*]]:2 = scf.for %[[EIV:.*]] = %[[C18]] to %[[C20]] step %[[C1]]
//  UNROLL-BY-3-SAME:     iter_args(%[[EARG0:.*]] = %[[FOR]]#0, %[[EARG1:.*]] = %[[FOR]]#1) -> (f32, f32)  : i32 {
//  UNROLL-BY-3-NEXT:     %[[EADD:.*]] = arith.addf %[[EARG0]], %[[EARG1]] : f32
//  UNROLL-BY-3-NEXT:     %[[EMUL:.*]] = arith.mulf %[[EARG0]], %[[EARG1]] : f32
//  UNROLL-BY-3-NEXT:     scf.yield %[[EADD]], %[[EMUL]] : f32, f32
//  UNROLL-BY-3-NEXT:   }
//  UNROLL-BY-3-NEXT:   return %[[EFOR]]#0, %[[EFOR]]#1 : f32, f32

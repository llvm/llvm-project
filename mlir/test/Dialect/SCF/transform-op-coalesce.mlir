// RUN: mlir-opt %s -transform-interpreter -split-input-file -verify-diagnostics -allow-unregistered-dialect --cse --mlir-print-local-scope | FileCheck %s

func.func @coalesce_inner() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  // CHECK: scf.for %[[IV0:.+]]
  // CHECK:   scf.for %[[IV1:.+]]
  // CHECK:     scf.for %[[IV2:.+]]
  // CHECK-NOT:   scf.for %[[IV3:.+]]
  scf.for %i = %c0 to %c10 step %c1 {
    scf.for %j = %c0 to %c10 step %c1 {
      scf.for %k = %i to %j step %c1 {
        // Inner loop must have been removed.
        scf.for %l = %i to %j step %c1 {
          "use"(%i, %j) : (index, index) -> ()
        }
      } {coalesce}
    }
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} attributes {coalesce} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.cast %0 : !transform.any_op to !transform.op<"scf.for">
    %2 = transform.loop.coalesce %1: (!transform.op<"scf.for">) -> (!transform.op<"scf.for">)
    transform.yield
  }
}

// -----

func.func @coalesce_outer(%arg1: memref<64x64xf32, 1>, %arg2: memref<64x64xf32, 1>, %arg3: memref<64x64xf32, 1>) attributes {} {
  // CHECK: %[[T0:.+]] = affine.apply affine_map<() -> (64)>()
  // CHECK: %[[UB:.+]] = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%[[T0]])[%[[T0]]]
  // CHECK: affine.for %[[IV1:.+]] = 0 to %[[UB:.+]] {
  // CHECK-NOT: affine.for %[[IV2:.+]]
  affine.for %arg4 = 0 to 64 {
    affine.for %arg5 = 0 to 64 {
      // CHECK: %[[IDX0:.+]] = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%[[IV1]])[%{{.+}}]
      // CHECK: %[[IDX1:.+]] = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%[[IV1]])[%{{.+}}]
      // CHECK-NEXT: %{{.+}} = affine.load %{{.+}}[%[[IDX1]], %[[IDX0]]] : memref<64x64xf32, 1>
      %0 = affine.load %arg1[%arg4, %arg5] : memref<64x64xf32, 1>
      %1 = affine.load %arg2[%arg4, %arg5] : memref<64x64xf32, 1>
      %2 = arith.addf %0, %1 : f32
      affine.store %2, %arg3[%arg4, %arg5] : memref<64x64xf32, 1>
    }
  } {coalesce}
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["affine.for"]} attributes {coalesce} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.cast %0 : !transform.any_op to !transform.op<"affine.for">
    %2 = transform.loop.coalesce %1 : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">)
    transform.yield
  }
}

// -----

func.func @coalesce_and_unroll(%arg1: memref<64x64xf32, 1>, %arg2: memref<64x64xf32, 1>, %arg3: memref<64x64xf32, 1>) attributes {} {
  // CHECK: scf.for %[[IV1:.+]] =
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index

  scf.for %arg4 = %c0 to %c64 step %c1 {
    // CHECK-NOT: scf.for
    scf.for %arg5 = %c0 to %c64 step %c1 {
      // CHECK: %[[IDX:.+]]:2 = affine.delinearize_index
      // CHECK-NEXT: %{{.+}} = memref.load %{{.+}}[%[[IDX]]#0, %[[IDX]]#1] : memref<64x64xf32, 1>
      %0 = memref.load %arg1[%arg4, %arg5] : memref<64x64xf32, 1>
      %1 = memref.load %arg2[%arg4, %arg5] : memref<64x64xf32, 1>
      %2 = arith.addf %0, %1 : f32
      // CHECK: memref.store
      // CHECK: memref.store
      // CHECK: memref.store
      // Residual loop must have a single store.
      // CHECK: memref.store
      memref.store %2, %arg3[%arg4, %arg5] : memref<64x64xf32, 1>
    }
  } {coalesce}
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} attributes {coalesce} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.cast %0 : !transform.any_op to !transform.op<"scf.for">
    %2 = transform.loop.coalesce %1 : (!transform.op<"scf.for">) -> (!transform.op<"scf.for">)
    transform.loop.unroll %2 {factor = 3} : !transform.op<"scf.for">
    transform.yield
  }
}

// -----

func.func @tensor_loops(%arg0 : tensor<?x?xf32>, %lb0 : index, %ub0 : index, %step0 : index,
    %lb1 : index, %ub1 : index, %step1 : index, %lb2 : index, %ub2 : index, %step2 : index) -> tensor<?x?xf32> {
  %0 = scf.for %i = %lb0 to %ub0 step %step0 iter_args(%arg1 = %arg0) -> tensor<?x?xf32> {
    %1 = scf.for %j = %lb1 to %ub1 step %step1 iter_args(%arg2 = %arg1) -> tensor<?x?xf32> {
      %2 = scf.for %k = %lb2 to %ub2 step %step2 iter_args(%arg3 = %arg2) -> tensor<?x?xf32> {
        %3 = "use"(%arg3, %i, %j, %k) : (tensor<?x?xf32>, index, index, index) -> (tensor<?x?xf32>)
        scf.yield %3 : tensor<?x?xf32>
      }
      scf.yield %2 : tensor<?x?xf32>
    }
    scf.yield %1 : tensor<?x?xf32>
  } {coalesce}
  return %0 : tensor<?x?xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} attributes {coalesce} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.cast %0 : !transform.any_op to !transform.op<"scf.for">
    %2 = transform.loop.coalesce %1 : (!transform.op<"scf.for">) -> (!transform.op<"scf.for">)
    transform.yield
  }
}
//      CHECK: func.func @tensor_loops(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[LB0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[UB0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[STEP0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[LB1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[UB1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[STEP1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[LB2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[UB2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[STEP2:[a-zA-Z0-9_]+]]: index
//      CHECK:   %[[NITERS0:.+]] = affine.apply
// CHECK-SAME:       affine_map<()[s0, s1, s2] -> ((-s0 + s1) ceildiv s2)>()[%[[LB0]], %[[UB0]], %[[STEP0]]]
//      CHECK:   %[[C0:.+]] = arith.constant 0 : index
//      CHECK:   %[[C1:.+]] = arith.constant 1 : index
//      CHECK:   %[[NITERS1:.+]] = affine.apply
// CHECK-SAME:       affine_map<()[s0, s1, s2] -> ((-s0 + s1) ceildiv s2)>()[%[[LB1]], %[[UB1]], %[[STEP1]]]
//      CHECK:   %[[NITERS2:.+]] = affine.apply
// CHECK-SAME:        affine_map<()[s0, s1, s2] -> ((-s0 + s1) ceildiv s2)>()[%[[LB2]], %[[UB2]], %[[STEP2]]]
//      CHECK:   %[[NEWUB:.+]] = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6, s7, s8] ->
// CHECK-SAME:       ((((-s0 + s1) ceildiv s2) * ((-s3 + s4) ceildiv s5)) * ((-s6 + s7) ceildiv s8))
// CHECK-SAME:       [%[[LB0]], %[[UB0]], %[[STEP0]], %[[LB1]], %[[UB1]], %[[STEP1]], %[[LB2]], %[[UB2]], %[[STEP2]]]
//      CHECK:   %[[RESULT:.+]] = scf.for %[[IV:[a-zA-Z0-9]+]] = %[[C0]] to %[[NEWUB]] step %[[C1]] iter_args(%[[ITER_ARG:.+]] = %[[ARG0]])
//      CHECK:     %[[DELINEARIZE:.+]]:3 = affine.delinearize_index %[[IV]] into (%[[NITERS0]], %[[NITERS1]], %[[NITERS2]])
//  CHECK-DAG:     %[[K:.+]] = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>(%[[DELINEARIZE]]#2)[%[[LB2]], %[[STEP2]]]
//  CHECK-DAG:     %[[J:.+]] = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>(%[[DELINEARIZE]]#1)[%[[LB1]], %[[STEP1]]]
//  CHECK-DAG:     %[[I:.+]] = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>(%[[DELINEARIZE]]#0)[%[[LB0]], %[[STEP0]]]
//      CHECK:     %[[USE:.+]] = "use"(%[[ITER_ARG]], %[[I]], %[[J]], %[[K]])
//      CHECK:     scf.yield %[[USE]]
//      CHECK:   return %[[RESULT]]

// -----

// Coalesce only first two loops, but not the last since the iter_args dont line up
func.func @tensor_loops_first_two(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %lb0 : index, %ub0 : index, %step0 : index,
    %lb1 : index, %ub1 : index, %step1 : index, %lb2 : index, %ub2 : index, %step2 : index) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %0:2 = scf.for %i = %lb0 to %ub0 step %step0 iter_args(%arg2 = %arg0, %arg3 = %arg1) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
    %1:2 = scf.for %j = %lb1 to %ub1 step %step1 iter_args(%arg4 = %arg2, %arg5 = %arg3) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
      %2:2 = scf.for %k = %lb2 to %ub2 step %step2 iter_args(%arg6 = %arg5, %arg7 = %arg4) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
        %3:2 = "use"(%arg3, %i, %j, %k) : (tensor<?x?xf32>, index, index, index) -> (tensor<?x?xf32>, tensor<?x?xf32>)
        scf.yield %3#0, %3#1 : tensor<?x?xf32>, tensor<?x?xf32>
      }
      scf.yield %2#0, %2#1 : tensor<?x?xf32>, tensor<?x?xf32>
    }
    scf.yield %1#0, %1#1 : tensor<?x?xf32>, tensor<?x?xf32>
  } {coalesce}
  return %0#0, %0#1 : tensor<?x?xf32>, tensor<?x?xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} attributes {coalesce} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.cast %0 : !transform.any_op to !transform.op<"scf.for">
    %2 = transform.loop.coalesce %1 : (!transform.op<"scf.for">) -> (!transform.op<"scf.for">)
    transform.yield
  }
}
//      CHECK: func.func @tensor_loops_first_two(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[LB0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[UB0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[STEP0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[LB1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[UB1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[STEP1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[LB2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[UB2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[STEP2:[a-zA-Z0-9_]+]]: index
//      CHECK:   scf.for
//      CHECK:     affine.delinearize_index
//      CHECK:     scf.for %{{[a-zA-Z0-9]+}} = %[[LB2]] to %[[UB2]] step %[[STEP2]]
//  CHECK-NOT:       scf.for
//      CHECK:   transform.named_sequence

// -----

// Coalesce only first two loops, but not the last since the yields dont match up
func.func @tensor_loops_first_two_2(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %lb0 : index, %ub0 : index, %step0 : index,
    %lb1 : index, %ub1 : index, %step1 : index, %lb2 : index, %ub2 : index, %step2 : index) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %0:2 = scf.for %i = %lb0 to %ub0 step %step0 iter_args(%arg2 = %arg0, %arg3 = %arg1) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
    %1:2 = scf.for %j = %lb1 to %ub1 step %step1 iter_args(%arg4 = %arg2, %arg5 = %arg3) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
      %2:2 = scf.for %k = %lb2 to %ub2 step %step2 iter_args(%arg6 = %arg4, %arg7 = %arg5) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
        %3:2 = "use"(%arg3, %i, %j, %k) : (tensor<?x?xf32>, index, index, index) -> (tensor<?x?xf32>, tensor<?x?xf32>)
        scf.yield %3#0, %3#1 : tensor<?x?xf32>, tensor<?x?xf32>
      }
      scf.yield %2#1, %2#0 : tensor<?x?xf32>, tensor<?x?xf32>
    }
    scf.yield %1#0, %1#1 : tensor<?x?xf32>, tensor<?x?xf32>
  } {coalesce}
  return %0#0, %0#1 : tensor<?x?xf32>, tensor<?x?xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} attributes {coalesce} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.cast %0 : !transform.any_op to !transform.op<"scf.for">
    %2 = transform.loop.coalesce %1 : (!transform.op<"scf.for">) -> (!transform.op<"scf.for">)
    transform.yield
  }
}
//      CHECK: func.func @tensor_loops_first_two_2(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[LB0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[UB0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[STEP0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[LB1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[UB1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[STEP1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[LB2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[UB2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[STEP2:[a-zA-Z0-9_]+]]: index
//      CHECK:   scf.for
//      CHECK:     affine.delinearize_index
//      CHECK:     scf.for %{{[a-zA-Z0-9]+}} = %[[LB2]] to %[[UB2]] step %[[STEP2]]
//  CHECK-NOT:       scf.for
//      CHECK:   transform.named_sequence

// -----

// Coalesce only last two loops, but not the first since the yields dont match up
func.func @tensor_loops_last_two(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %lb0 : index, %ub0 : index, %step0 : index,
    %lb1 : index, %ub1 : index, %step1 : index, %lb2 : index, %ub2 : index, %step2 : index) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %0:2 = scf.for %i = %lb0 to %ub0 step %step0 iter_args(%arg2 = %arg0, %arg3 = %arg1) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
    %1:2 = scf.for %j = %lb1 to %ub1 step %step1 iter_args(%arg4 = %arg2, %arg5 = %arg3) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
      %2:2 = scf.for %k = %lb2 to %ub2 step %step2 iter_args(%arg6 = %arg4, %arg7 = %arg5) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
        %3:2 = "use"(%arg3, %i, %j, %k) : (tensor<?x?xf32>, index, index, index) -> (tensor<?x?xf32>, tensor<?x?xf32>)
        scf.yield %3#0, %3#1 : tensor<?x?xf32>, tensor<?x?xf32>
      }
      scf.yield %2#0, %2#1 : tensor<?x?xf32>, tensor<?x?xf32>
    }
    scf.yield %1#1, %1#0 : tensor<?x?xf32>, tensor<?x?xf32>
  } {coalesce}
  return %0#0, %0#1 : tensor<?x?xf32>, tensor<?x?xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} attributes {coalesce} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.cast %0 : !transform.any_op to !transform.op<"scf.for">
    %2 = transform.loop.coalesce %1 : (!transform.op<"scf.for">) -> (!transform.op<"scf.for">)
    transform.yield
  }
}
//      CHECK: func.func @tensor_loops_last_two(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[LB0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[UB0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[STEP0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[LB1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[UB1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[STEP1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[LB2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[UB2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[STEP2:[a-zA-Z0-9_]+]]: index
//      CHECK:   scf.for %{{[a-zA-Z0-9]+}} = %[[LB0]] to %[[UB0]] step %[[STEP0]]
//  CHECK-NOT:     affine.delinearize_index
//      CHECK:     scf.for
//      CHECK:       affine.delinearize_index
//  CHECK-NOT:       scf.for
//      CHECK:   transform.named_sequence

// -----

// Check avoiding generating unnecessary operations while collapsing trip-1 loops.
func.func @trip_one_loops(%arg0 : tensor<?x?xf32>, %arg1 : index, %arg2 : index) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = scf.for %iv0 = %c0 to %c1 step %c1 iter_args(%iter0 = %arg0) -> tensor<?x?xf32> {
    %1 = scf.for %iv1 = %c0 to %c1 step %c1 iter_args(%iter1 = %iter0) -> tensor<?x?xf32> {
      %2 = scf.for %iv2 = %c0 to %arg1 step %c1 iter_args(%iter2 = %iter1) -> tensor<?x?xf32> {
        %3 = scf.for %iv3 = %c0 to %c1 step %c1 iter_args(%iter3 = %iter2) -> tensor<?x?xf32> {
          %4 = scf.for %iv4 = %c0 to %arg2 step %c1 iter_args(%iter4 = %iter3) -> tensor<?x?xf32> {
            %5 = "some_use"(%iter4, %iv0, %iv1, %iv2, %iv3, %iv4)
              : (tensor<?x?xf32>, index, index, index, index, index) -> (tensor<?x?xf32>)
            scf.yield %5 : tensor<?x?xf32>
          }
          scf.yield %4 : tensor<?x?xf32>
        }
        scf.yield %3 : tensor<?x?xf32>
      }
      scf.yield %2 : tensor<?x?xf32>
    }
    scf.yield %1 : tensor<?x?xf32>
  } {coalesce}
  return %0 : tensor<?x?xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} attributes {coalesce} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.cast %0 : !transform.any_op to !transform.op<"scf.for">
    %2 = transform.loop.coalesce %1 : (!transform.op<"scf.for">) -> (!transform.op<"scf.for">)
    transform.apply_patterns to %2 {
      transform.apply_patterns.canonicalization
    } : !transform.op<"scf.for">
    transform.yield
  }
}
// CHECK-LABEL: func @trip_one_loops
//  CHECK-SAME:     , %[[ARG1:.+]]: index,
//  CHECK-SAME:     %[[ARG2:.+]]: index)
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//       CHECK:   %[[UB:.+]] = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%[[ARG1]], %[[ARG2]]]
//       CHECK:   scf.for %[[IV:.+]] = %[[C0]] to %[[UB]] step %[[C1]]
//       CHECK:     %[[DELINEARIZE:.+]]:2 = affine.delinearize_index %[[IV]] into (%[[ARG1]], %[[ARG2]])
//       CHECK:     "some_use"(%{{[a-zA-Z0-9]+}}, %[[C0]], %[[C0]], %[[DELINEARIZE]]#0, %[[C0]], %[[DELINEARIZE]]#1)

// -----

// Check generating no instructions when all except one loops is non unit-trip.
func.func @all_outer_trip_one(%arg0 : tensor<?x?xf32>, %arg1 : index) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = scf.for %iv0 = %c0 to %c1 step %c1 iter_args(%iter0 = %arg0) -> tensor<?x?xf32> {
    %1 = scf.for %iv1 = %c0 to %c1 step %c1 iter_args(%iter1 = %iter0) -> tensor<?x?xf32> {
      %2 = scf.for %iv2 = %c0 to %arg1 step %c1 iter_args(%iter2 = %iter1) -> tensor<?x?xf32> {
        %3 = "some_use"(%iter2, %iv0, %iv1, %iv2)
          : (tensor<?x?xf32>, index, index, index) -> (tensor<?x?xf32>)
        scf.yield %3 : tensor<?x?xf32>
      }
      scf.yield %2 : tensor<?x?xf32>
    }
    scf.yield %1 : tensor<?x?xf32>
  } {coalesce}
  return %0 : tensor<?x?xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} attributes {coalesce} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.cast %0 : !transform.any_op to !transform.op<"scf.for">
    %2 = transform.loop.coalesce %1 : (!transform.op<"scf.for">) -> (!transform.op<"scf.for">)
    transform.apply_patterns to %2 {
      transform.apply_patterns.canonicalization
    } : !transform.op<"scf.for">
    transform.yield
  }
}
// CHECK-LABEL: func @all_outer_trip_one
//  CHECK-SAME:     , %[[ARG1:.+]]: index)
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//       CHECK:   scf.for %[[IV:.+]] = %[[C0]] to %[[ARG1]] step %[[C1]]
//       CHECK:     "some_use"(%{{[a-zA-Z0-9]+}}, %[[C0]], %[[C0]], %[[IV]])

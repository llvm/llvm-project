// RUN: mlir-opt %s -test-scf-for-op-dead-cycles -canonicalize -split-input-file | FileCheck %s

// CHECK-LABEL: func @dead_arg(
//  CHECK-SAME:   %[[A0:.*]]: f32, %[[A1:.*]]: f32)
//       CHECK:   scf.for {{.*}} iter_args(%[[ARG1:.*]] = %[[A1]])
//  CHECK-NEXT:     %[[S:.+]] = arith.addf %[[ARG1]], %[[ARG1]] : f32
//  CHECK-NEXT:     scf.yield %[[S]]
func.func @dead_arg(%a0 : f32, %a1 : f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0, %1 = scf.for %i = %c0 to %c10 step %c1
  iter_args(%arg0 = %a0, %arg1 = %a1) -> (f32, f32) {
    %s0 = arith.addf %arg0, %arg0 : f32
    %s1 = arith.addf %arg1, %arg1 : f32
    scf.yield %s0, %s1 : f32, f32
  }
  return %1 : f32
}

// -----

// CHECK-LABEL: func @dead_arg_negative(
//  CHECK-SAME:   %[[A0:.*]]: f32, %[[A1:.*]]: f32)
//       CHECK:   scf.for {{.*}} iter_args(%[[ARG0:.*]] = %[[A0]], %[[ARG1:.*]] = %[[A1]])
//  CHECK-NEXT:     %[[S0:.+]] = arith.addf %[[ARG0]], %[[ARG0]] : f32
//  CHECK-NEXT:     %[[S1:.+]] = arith.addf %[[ARG1]], %[[ARG1]] : f32
//  CHECK-NEXT:     scf.yield %[[S1]], %[[S0]]
func.func @dead_arg_negative(%a0 : f32, %a1 : f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0, %1 = scf.for %i = %c0 to %c10 step %c1
  iter_args(%arg0 = %a0, %arg1 = %a1) -> (f32, f32) {
    %s0 = arith.addf %arg0, %arg0 : f32
    %s1 = arith.addf %arg1, %arg1 : f32
    scf.yield %s1, %s0 : f32, f32
  }
  return %1 : f32
}

// -----

// CHECK-LABEL: func @dead_arg_side_effect(
//  CHECK-SAME:   %[[A:.*]]: f32
//       CHECK:   scf.for {{.*}} iter_args(%[[ARG0:.*]] = %[[A]])
//  CHECK-NEXT:     %[[S0:.+]] = arith.addf %[[ARG0]], %[[ARG0]] : f32
//  CHECK-NEXT:     memref.store
//  CHECK-NEXT:     scf.yield %[[S0]]
func.func @dead_arg_side_effect(%a : f32, %A : memref<f32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0, %1 = scf.for %i = %c0 to %c10 step %c1
  iter_args(%arg0 = %a, %arg1 = %a) -> (f32, f32) {
    %s0 = arith.addf %arg0, %arg0 : f32
    %s1 = arith.addf %arg1, %arg1 : f32
    memref.store %s0, %A[]: memref<f32>
    scf.yield %s0, %s1 : f32, f32
  }
  return
}

// -----

// CHECK-LABEL: func @dead_arg_recurse(
//  CHECK-SAME:   %[[A0:.*]]: f32, %[[A1:.*]]: f32, %[[A2:.*]]: f32, %[[A3:.*]]: f32)
//       CHECK:   scf.for {{.*}} iter_args(%[[ARG1:.*]] = %[[A1]], %[[ARG3:.*]] = %[[A3]])
//  CHECK-NEXT:     %[[S0:.+]] = arith.addf %[[ARG1]], %[[ARG1]] : f32
//  CHECK-NEXT:     %[[S1:.+]] = arith.addf %[[ARG1]], %[[ARG3]] : f32
//  CHECK-NEXT:     scf.yield %[[S0]], %[[S1]]
func.func @dead_arg_recurse(%a0 : f32, %a1 : f32, %a2 : f32, %a3 : f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0:4 = scf.for %i = %c0 to %c10 step %c1
  iter_args(%arg0 = %a0, %arg1 = %a1, %arg2 = %a2, %arg3 = %a3) -> (f32, f32, f32, f32) {
    %s0 = arith.addf %arg0, %arg3 : f32
    %s1 = arith.addf %arg1, %arg1 : f32
    %s2 = arith.addf %arg2, %arg0 : f32
    %s3 = arith.addf %arg1, %arg3 : f32
    scf.yield %s0, %s1, %s2, %s3 : f32, f32, f32, f32
  }
  return %0#3 : f32
}

// -----

// CHECK-LABEL: func @dead_arg_nested(
//  CHECK-SAME:   %[[A0:.*]]: f32, %[[A1:.*]]: f32)
//       CHECK:   scf.for {{.*}} iter_args(%[[ARG1:.*]] = %[[A1]])
//       CHECK:     %[[R:.+]] = scf.for {{.*}} iter_args(%[[ARG4:.*]] = %[[ARG1]])
//  CHECK-NEXT:       %[[S:.+]] = arith.addf %[[ARG4]], %[[ARG4]] : f32
//  CHECK-NEXT:       scf.yield %[[S]]
//  CHECK-NEXT:     }
//  CHECK-NEXT:     scf.yield %[[R]]
func.func @dead_arg_nested(%a0 : f32, %a1 : f32) -> f32{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0:2 = scf.for %i = %c0 to %c10 step %c1
  iter_args(%arg0 = %a0, %arg1 = %a1) -> (f32, f32) {
    %1:2 = scf.for %i1 = %c0 to %c10 step %c1
    iter_args(%arg4 = %arg0, %arg5 = %arg1) -> (f32, f32) {
      %s1 = arith.addf %arg4, %arg4 : f32
      %s2 = arith.addf %arg5, %arg5 : f32
      scf.yield %s1, %s2 : f32, f32
    }
    scf.yield %1#0, %1#1 : f32, f32
  }
  return %0#1 : f32
}

// -----

// CHECK-LABEL: func @dead_arg_nested_if(
//  CHECK-SAME:   %[[A0:.*]]: f32, %[[A1:.*]]: f32, %{{.*}}: i1)
//       CHECK:   scf.for {{.*}} iter_args(%[[ARG1:.*]] = %[[A1]])
//       CHECK:     %[[R:.+]] = scf.if {{.*}} {
//  CHECK-NEXT:       %[[S:.+]] = arith.addf %[[ARG1]], %[[ARG1]] : f32
//  CHECK-NEXT:       scf.yield %[[S]]
//  CHECK-NEXT:     } else {
//  CHECK-NEXT:       scf.yield %{{.*}} : f32
//  CHECK-NEXT:     }
//  CHECK-NEXT:     scf.yield %[[R]]
func.func @dead_arg_nested_if(%a0 : f32, %a1 : f32, %c: i1) -> f32{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0:2 = scf.for %i = %c0 to %c10 step %c1
  iter_args(%arg0 = %a0, %arg1 = %a1) -> (f32, f32) {
    %1:2 = scf.if %c -> (f32, f32) {
      %s1 = arith.addf %arg0, %arg0 : f32
      %s2 = arith.addf %arg1, %arg1 : f32
      scf.yield %s1, %s2 : f32, f32
    } else {
      %cst_0 = arith.constant 1.000000e+00 : f32
      %cst_1 = arith.constant 2.000000e+00 : f32
      scf.yield %cst_0, %cst_1 : f32, f32
    }
    scf.yield %1#0, %1#1 : f32, f32
  }
  return %0#1 : f32
}


// RUN: mlir-opt -allow-unregistered-dialect -test-scf-for-utils=test-replace-with-new-yields -mlir-disable-threading %s | FileCheck %s

func.func @doubleup(%lb: index, %ub: index, %step: index, %extra_arg: f32) -> f32 {
  %0 = scf.for %i = %lb to %ub step %step iter_args(%iter = %extra_arg) -> (f32) {
    %1 = arith.addf %iter, %iter : f32
    scf.yield %1: f32
  }
  return %0: f32
}
// CHECK-LABEL: func @doubleup
//  CHECK-SAME:   %[[ARG:[a-zA-Z0-9]+]]: f32
//       CHECK:   %[[NEWLOOP:.+]]:2 = scf.for
//  CHECK-SAME:       iter_args(%[[INIT1:.+]] = %[[ARG]], %[[INIT2:.+]] = %[[ARG]]
//       CHECK:     %[[DOUBLE:.+]] = arith.addf %[[INIT1]], %[[INIT1]]
//       CHECK:     %[[DOUBLE2:.+]] = arith.addf %[[DOUBLE]], %[[DOUBLE]]
//       CHECK:     scf.yield %[[DOUBLE]], %[[DOUBLE2]]
//       CHECK:   return %[[NEWLOOP]]#0

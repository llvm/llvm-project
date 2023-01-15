// RUN: mlir-opt %s --split-input-file -test-expand-math | FileCheck %s

// CHECK-LABEL: func @tanh
func.func @tanh(%arg: f32) -> f32 {
  %res = math.tanh %arg : f32
  return %res : f32
}
// CHECK-DAG: %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[ONE:.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG: %[[TWO:.+]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[DOUBLEDX:.+]] = arith.mulf %arg0, %[[TWO]] : f32
// CHECK: %[[NEGDOUBLEDX:.+]] = arith.negf %[[DOUBLEDX]] : f32
// CHECK: %[[EXP1:.+]] = math.exp %[[NEGDOUBLEDX]] : f32
// CHECK: %[[DIVIDEND1:.+]] = arith.subf %[[ONE]], %[[EXP1]] : f32
// CHECK: %[[DIVISOR1:.+]] = arith.addf %[[EXP1]], %[[ONE]] : f32
// CHECK: %[[RES1:.+]] = arith.divf %[[DIVIDEND1]], %[[DIVISOR1]] : f32
// CHECK: %[[EXP2:.+]] = math.exp %[[DOUBLEDX]] : f32
// CHECK: %[[DIVIDEND2:.+]] = arith.subf %[[EXP2]], %[[ONE]] : f32
// CHECK: %[[DIVISOR2:.+]] = arith.addf %[[EXP2]], %[[ONE]] : f32
// CHECK: %[[RES2:.+]] = arith.divf %[[DIVIDEND2]], %[[DIVISOR2]] : f32
// CHECK: %[[COND:.+]] = arith.cmpf oge, %arg0, %[[ZERO]] : f32
// CHECK: %[[RESULT:.+]] = arith.select %[[COND]], %[[RES1]], %[[RES2]] : f32
// CHECK: return %[[RESULT]]

// -----

// CHECK-LABEL: func @ctlz
func.func @ctlz(%arg: i32) -> i32 {
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i32
  // CHECK-DAG: %[[C32:.+]] = arith.constant 32 : i32
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i32
  // CHECK: %[[WHILE:.+]]:3 = scf.while (%[[A1:.+]] = %arg0, %[[A2:.+]] = %[[C32]], %[[A3:.+]] = %[[C0]])
  // CHECK:   %[[CMP:.+]] = arith.cmpi ne, %[[A1]], %[[A3]]
  // CHECK:   scf.condition(%[[CMP]]) %[[A1]], %[[A2]], %[[A3]]
  // CHECK:   %[[SHR:.+]] = arith.shrui %[[A1]], %[[C1]]
  // CHECK:   %[[SUB:.+]] = arith.subi %[[A2]], %[[C1]]
  // CHECK:   scf.yield %[[SHR]], %[[SUB]], %[[A3]]
  %res = math.ctlz %arg : i32

  // CHECK: return %[[WHILE]]#1
  return %res : i32
}

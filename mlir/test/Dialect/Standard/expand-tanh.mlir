// RUN: mlir-opt %s -test-expand-tanh | FileCheck %s

// CHECK-LABEL: func @tanh
func @tanh(%arg: f32) -> f32 {
  %res = math.tanh %arg : f32
  return %res : f32
}
// CHECK-DAG: %[[ZERO:.+]] = constant 0.000000e+00 : f32
// CHECK-DAG: %[[ONE:.+]] = constant 1.000000e+00 : f32
// CHECK-DAG: %[[TWO:.+]] = constant 2.000000e+00 : f32
// CHECK: %[[DOUBLEDX:.+]] = mulf %arg0, %[[TWO]] : f32
// CHECK: %[[NEGDOUBLEDX:.+]] = negf %[[DOUBLEDX]] : f32
// CHECK: %[[EXP1:.+]] = math.exp %[[NEGDOUBLEDX]] : f32
// CHECK: %[[DIVIDEND1:.+]] = subf %[[ONE]], %[[EXP1]] : f32
// CHECK: %[[DIVISOR1:.+]] = addf %[[ONE]], %[[EXP1]] : f32
// CHECK: %[[RES1:.+]] = divf %[[DIVIDEND1]], %[[DIVISOR1]] : f32
// CHECK: %[[EXP2:.+]] = math.exp %[[DOUBLEDX]] : f32
// CHECK: %[[DIVIDEND2:.+]] = subf %[[EXP2]], %[[ONE]] : f32
// CHECK: %[[DIVISOR2:.+]] = addf %[[EXP2]], %[[ONE]] : f32
// CHECK: %[[RES2:.+]] = divf %[[DIVIDEND2]], %[[DIVISOR2]] : f32
// CHECK: %[[COND:.+]] = cmpf oge, %arg0, %[[ZERO]] : f32
// CHECK: %[[RESULT:.+]] = select %[[COND]], %[[RES1]], %[[RES2]] : f32
// CHECK: return %[[RESULT]]

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


// CHECK-LABEL: func @vector_tanh
func.func @vector_tanh(%arg: vector<4xf32>) -> vector<4xf32> {
  // CHECK-NOT: math.tanh
  %res = math.tanh %arg : vector<4xf32>
  return %res : vector<4xf32>
}

// -----

// CHECK-LABEL: func @tan
func.func @tan(%arg: f32) -> f32 {
  %res = math.tan %arg : f32
  return %res : f32
}

// CHECK-SAME: %[[ARG0:.+]]: f32
// CHECK: %[[SIN:.+]] = math.sin %[[ARG0]]
// CHECK: %[[COS:.+]] = math.cos %[[ARG0]]
// CHECK: %[[DIV:.+]] = arith.divf %[[SIN]], %[[COS]]


// -----

// CHECK-LABEL: func @vector_tan
func.func @vector_tan(%arg: vector<4xf32>) -> vector<4xf32> {
  %res = math.tan %arg : vector<4xf32>
  return %res : vector<4xf32>
}

// CHECK-NOT: math.tan

// -----

func.func @ctlz(%arg: i32) -> i32 {
  %res = math.ctlz %arg : i32
  return %res : i32
}

// CHECK-LABEL: @ctlz
// CHECK-SAME: %[[ARG0:.+]]: i32
// CHECK-DAG: %[[C0:.+]] = arith.constant 0
// CHECK-DAG: %[[C16:.+]] = arith.constant 16
// CHECK-DAG: %[[C65535:.+]] = arith.constant 65535
// CHECK-DAG: %[[C8:.+]] = arith.constant 8
// CHECK-DAG: %[[C16777215:.+]] = arith.constant 16777215
// CHECK-DAG: %[[C4:.+]] = arith.constant 4
// CHECK-DAG: %[[C268435455:.+]] = arith.constant 268435455
// CHECK-DAG: %[[C2:.+]] = arith.constant 2
// CHECK-DAG: %[[C1073741823:.+]] = arith.constant 1073741823
// CHECK-DAG: %[[C1:.+]] = arith.constant 1
// CHECK-DAG: %[[C2147483647:.+]] = arith.constant 2147483647
// CHECK-DAG: %[[C32:.+]] = arith.constant 32

// CHECK: %[[PRED:.+]] = arith.cmpi ule, %[[ARG0]], %[[C65535]]
// CHECK: %[[SHL:.+]] = arith.shli %[[ARG0]], %[[C16]]
// CHECK: %[[SELX0:.+]] = arith.select %[[PRED]], %[[SHL]], %[[ARG0]]
// CHECK: %[[SELY0:.+]] = arith.select %[[PRED]], %[[C16]], %[[C0]]

// CHECK: %[[PRED:.+]] = arith.cmpi ule, %[[SELX0]], %[[C16777215]]
// CHECK: %[[ADD:.+]] = arith.addi %[[SELY0]], %[[C8]]
// CHECK: %[[SHL:.+]] = arith.shli %[[SELX0]], %[[C8]]
// CHECK: %[[SELX1:.+]] = arith.select %[[PRED]], %[[SHL]], %[[SELX0]]
// CHECK: %[[SELY1:.+]] = arith.select %[[PRED]], %[[ADD]], %[[SELY0]]

// CHECK: %[[PRED:.+]] = arith.cmpi ule, %[[SELX1]], %[[C268435455]] : i32
// CHECK: %[[ADD:.+]] = arith.addi %[[SELY1]], %[[C4]]
// CHECK: %[[SHL:.+]] = arith.shli %[[SELX1]], %[[C4]]
// CHECK: %[[SELX2:.+]] = arith.select %[[PRED]], %[[SHL]], %[[SELX1]]
// CHECK: %[[SELY2:.+]] = arith.select %[[PRED]], %[[ADD]], %[[SELY1]]


// CHECK: %[[PRED:.+]] = arith.cmpi ule, %[[SELX2]], %[[C1073741823]] : i32
// CHECK: %[[ADD:.+]] = arith.addi %[[SELY2]], %[[C2]]
// CHECK: %[[SHL:.+]] = arith.shli %[[SELX2]], %[[C2]]
// CHECK: %[[SELX3:.+]] = arith.select %[[PRED]], %[[SHL]], %[[SELX2]]
// CHECK: %[[SELY3:.+]] = arith.select %[[PRED]], %[[ADD]], %[[SELY2]]

// CHECK: %[[PRED:.+]] = arith.cmpi ule, %[[SELX3]], %[[C2147483647]] : i32
// CHECK: %[[ADD:.+]] = arith.addi %[[SELY3]], %[[C1]]
// CHECK: %[[SELY4:.+]] = arith.select %[[PRED]], %[[ADD]], %[[SELY3]]

// CHECK: %[[PRED:.+]] = arith.cmpi eq, %[[ARG0]], %[[C0]] : i32
// CHECK: %[[SEL:.+]] = arith.select %[[PRED]], %[[C32]], %[[SELY4]] : i32
// CHECK: return %[[SEL]]

// -----

func.func @ctlz_vector(%arg: vector<4xi32>) -> vector<4xi32> {
  %res = math.ctlz %arg : vector<4xi32>
  return %res : vector<4xi32>
}

// CHECK-LABEL: @ctlz_vector
// CHECK-NOT: math.ctlz

// -----

// CHECK-LABEL:    func @fmaf_func
// CHECK-SAME:     ([[ARG0:%.+]]: f64, [[ARG1:%.+]]: f64, [[ARG2:%.+]]: f64) -> f64
func.func @fmaf_func(%a: f64, %b: f64, %c: f64) -> f64 {
  // CHECK-NEXT:     [[MULF:%.+]] = arith.mulf [[ARG0]], [[ARG1]]
  // CHECK-NEXT:     [[ADDF:%.+]] = arith.addf [[MULF]], [[ARG2]]
  // CHECK-NEXT:     return [[ADDF]]
  %ret = math.fma %a, %b, %c : f64
  return %ret : f64
}

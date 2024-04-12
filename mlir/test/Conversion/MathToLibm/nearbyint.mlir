// RUN: mlir-opt %s --pass-pipeline='builtin.module(convert-math-to-libm{allow-c23-features=0 rounding-mode-is-default}, canonicalize)' | FileCheck %s

// CHECK-DAG: @nearbyint(f64) -> f64 attributes {libm, llvm.readnone}
// CHECK-DAG: @nearbyintf(f32) -> f32 attributes {libm, llvm.readnone}

// CHECK-LABEL: func @nearbyint_caller
// CHECK-SAME: %[[FLOAT:.*]]: f32
// CHECK-SAME: %[[DOUBLE:.*]]: f64
func.func @nearbyint_caller(%float: f32, %double: f64) -> (f32, f64)  {
  // CHECK-DAG: %[[FLOAT_RESULT:.*]] = call @nearbyintf(%[[FLOAT]]) : (f32) -> f32
  %float_result = math.roundeven %float : f32
  // CHECK-DAG: %[[DOUBLE_RESULT:.*]] = call @nearbyint(%[[DOUBLE]]) : (f64) -> f64
  %double_result = math.roundeven %double : f64
  // CHECK: return %[[FLOAT_RESULT]], %[[DOUBLE_RESULT]]
  return %float_result, %double_result : f32, f64
}

// CHECK-LABEL:   func @nearbyint_vec_caller(
// CHECK-SAME:                           %[[VAL_0:.*]]: vector<2xf32>,
// CHECK-SAME:                           %[[VAL_1:.*]]: vector<2xf64>) -> (vector<2xf32>, vector<2xf64>) {
// CHECK-DAG:       %[[CVF:.*]] = arith.constant dense<0.000000e+00> : vector<2xf32>
// CHECK-DAG:       %[[CVD:.*]] = arith.constant dense<0.000000e+00> : vector<2xf64>
// CHECK:           %[[IN0_F32:.*]] = vector.extract %[[VAL_0]][0] : f32 from vector<2xf32>
// CHECK:           %[[OUT0_F32:.*]] = call @nearbyintf(%[[IN0_F32]]) : (f32) -> f32
// CHECK:           %[[VAL_8:.*]] = vector.insert %[[OUT0_F32]], %[[CVF]] [0] : f32 into vector<2xf32>
// CHECK:           %[[IN1_F32:.*]] = vector.extract %[[VAL_0]][1] : f32 from vector<2xf32>
// CHECK:           %[[OUT1_F32:.*]] = call @nearbyintf(%[[IN1_F32]]) : (f32) -> f32
// CHECK:           %[[VAL_11:.*]] = vector.insert %[[OUT1_F32]], %[[VAL_8]] [1] : f32 into vector<2xf32>
// CHECK:           %[[IN0_F64:.*]] = vector.extract %[[VAL_1]][0] : f64 from vector<2xf64>
// CHECK:           %[[OUT0_F64:.*]] = call @nearbyint(%[[IN0_F64]]) : (f64) -> f64
// CHECK:           %[[VAL_14:.*]] = vector.insert %[[OUT0_F64]], %[[CVD]] [0] : f64 into vector<2xf64>
// CHECK:           %[[IN1_F64:.*]] = vector.extract %[[VAL_1]][1] : f64 from vector<2xf64>
// CHECK:           %[[OUT1_F64:.*]] = call @nearbyint(%[[IN1_F64]]) : (f64) -> f64
// CHECK:           %[[VAL_17:.*]] = vector.insert %[[OUT1_F64]], %[[VAL_14]] [1] : f64 into vector<2xf64>
// CHECK:           return %[[VAL_11]], %[[VAL_17]] : vector<2xf32>, vector<2xf64>
// CHECK:         }
func.func @nearbyint_vec_caller(%float: vector<2xf32>, %double: vector<2xf64>) -> (vector<2xf32>, vector<2xf64>) {
  %float_result = math.roundeven %float : vector<2xf32>
  %double_result = math.roundeven %double : vector<2xf64>
  return %float_result, %double_result : vector<2xf32>, vector<2xf64>
}

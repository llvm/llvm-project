// RUN: mlir-opt %s -convert-math-to-libm -canonicalize | FileCheck %s

// CHECK-DAG: @erf(f64) -> f64
// CHECK-DAG: @erff(f32) -> f32
// CHECK-DAG: @expm1(f64) -> f64
// CHECK-DAG: @expm1f(f32) -> f32
// CHECK-DAG: @atan2(f64, f64) -> f64
// CHECK-DAG: @atan2f(f32, f32) -> f32
// CHECK-DAG: @tanh(f64) -> f64
// CHECK-DAG: @tanhf(f32) -> f32

// CHECK-LABEL: func @tanh_caller
// CHECK-SAME: %[[FLOAT:.*]]: f32
// CHECK-SAME: %[[DOUBLE:.*]]: f64
func @tanh_caller(%float: f32, %double: f64) -> (f32, f64)  {
  // CHECK-DAG: %[[FLOAT_RESULT:.*]] = call @tanhf(%[[FLOAT]]) : (f32) -> f32
  %float_result = math.tanh %float : f32
  // CHECK-DAG: %[[DOUBLE_RESULT:.*]] = call @tanh(%[[DOUBLE]]) : (f64) -> f64
  %double_result = math.tanh %double : f64
  // CHECK: return %[[FLOAT_RESULT]], %[[DOUBLE_RESULT]]
  return %float_result, %double_result : f32, f64
}


// CHECK-LABEL: func @atan2_caller
// CHECK-SAME: %[[FLOAT:.*]]: f32
// CHECK-SAME: %[[DOUBLE:.*]]: f64
func @atan2_caller(%float: f32, %double: f64) -> (f32, f64) {
  // CHECK-DAG: %[[FLOAT_RESULT:.*]] = call @atan2f(%[[FLOAT]], %[[FLOAT]]) : (f32, f32) -> f32
  %float_result = math.atan2 %float, %float : f32
  // CHECK-DAG: %[[DOUBLE_RESULT:.*]] = call @atan2(%[[DOUBLE]], %[[DOUBLE]]) : (f64, f64) -> f64
  %double_result = math.atan2 %double, %double : f64
  // CHECK: return %[[FLOAT_RESULT]], %[[DOUBLE_RESULT]]
  return %float_result, %double_result : f32, f64
}

// CHECK-LABEL: func @erf_caller
// CHECK-SAME: %[[FLOAT:.*]]: f32
// CHECK-SAME: %[[DOUBLE:.*]]: f64
func @erf_caller(%float: f32, %double: f64) -> (f32, f64)  {
  // CHECK-DAG: %[[FLOAT_RESULT:.*]] = call @erff(%[[FLOAT]]) : (f32) -> f32
  %float_result = math.erf %float : f32
  // CHECK-DAG: %[[DOUBLE_RESULT:.*]] = call @erf(%[[DOUBLE]]) : (f64) -> f64
  %double_result = math.erf %double : f64
  // CHECK: return %[[FLOAT_RESULT]], %[[DOUBLE_RESULT]]
  return %float_result, %double_result : f32, f64
}

// CHECK-LABEL: func @expm1_caller
// CHECK-SAME: %[[FLOAT:.*]]: f32
// CHECK-SAME: %[[DOUBLE:.*]]: f64
func @expm1_caller(%float: f32, %double: f64) -> (f32, f64) {
  // CHECK-DAG: %[[FLOAT_RESULT:.*]] = call @expm1f(%[[FLOAT]]) : (f32) -> f32
  %float_result = math.expm1 %float : f32
  // CHECK-DAG: %[[DOUBLE_RESULT:.*]] = call @expm1(%[[DOUBLE]]) : (f64) -> f64
  %double_result = math.expm1 %double : f64
  // CHECK: return %[[FLOAT_RESULT]], %[[DOUBLE_RESULT]]
  return %float_result, %double_result : f32, f64
}

func @expm1_vec_caller(%float: vector<2xf32>, %double: vector<2xf64>) -> (vector<2xf32>, vector<2xf64>) {
  %float_result = math.expm1 %float : vector<2xf32>
  %double_result = math.expm1 %double : vector<2xf64>
  return %float_result, %double_result : vector<2xf32>, vector<2xf64>
}
// CHECK-LABEL:   func @expm1_vec_caller(
// CHECK-SAME:                           %[[VAL_0:.*]]: vector<2xf32>,
// CHECK-SAME:                           %[[VAL_1:.*]]: vector<2xf64>) -> (vector<2xf32>, vector<2xf64>) {
// CHECK-DAG:       %[[CVF:.*]] = arith.constant dense<0.000000e+00> : vector<2xf32>
// CHECK-DAG:       %[[CVD:.*]] = arith.constant dense<0.000000e+00> : vector<2xf64>
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : i32
// CHECK:           %[[IN0_F32:.*]] = vector.extractelement %[[VAL_0]]{{\[}}%[[C0]] : i32] : vector<2xf32>
// CHECK:           %[[OUT0_F32:.*]] = call @expm1f(%[[IN0_F32]]) : (f32) -> f32
// CHECK:           %[[VAL_8:.*]] = vector.insertelement %[[OUT0_F32]], %[[CVF]]{{\[}}%[[C0]] : i32] : vector<2xf32>
// CHECK:           %[[IN1_F32:.*]] = vector.extractelement %[[VAL_0]]{{\[}}%[[C1]] : i32] : vector<2xf32>
// CHECK:           %[[OUT1_F32:.*]] = call @expm1f(%[[IN1_F32]]) : (f32) -> f32
// CHECK:           %[[VAL_11:.*]] = vector.insertelement %[[OUT1_F32]], %[[VAL_8]]{{\[}}%[[C1]] : i32] : vector<2xf32>
// CHECK:           %[[IN0_F64:.*]] = vector.extractelement %[[VAL_1]]{{\[}}%[[C0]] : i32] : vector<2xf64>
// CHECK:           %[[OUT0_F64:.*]] = call @expm1(%[[IN0_F64]]) : (f64) -> f64
// CHECK:           %[[VAL_14:.*]] = vector.insertelement %[[OUT0_F64]], %[[CVD]]{{\[}}%[[C0]] : i32] : vector<2xf64>
// CHECK:           %[[IN1_F64:.*]] = vector.extractelement %[[VAL_1]]{{\[}}%[[C1]] : i32] : vector<2xf64>
// CHECK:           %[[OUT1_F64:.*]] = call @expm1(%[[IN1_F64]]) : (f64) -> f64
// CHECK:           %[[VAL_17:.*]] = vector.insertelement %[[OUT1_F64]], %[[VAL_14]]{{\[}}%[[C1]] : i32] : vector<2xf64>
// CHECK:           return %[[VAL_11]], %[[VAL_17]] : vector<2xf32>, vector<2xf64>
// CHECK:         }


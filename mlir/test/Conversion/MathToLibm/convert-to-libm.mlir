// RUN: mlir-opt %s -convert-math-to-libm -canonicalize | FileCheck %s

// CHECK-DAG: @acos(f64) -> f64 attributes {llvm.readnone}
// CHECK-DAG: @acosf(f32) -> f32 attributes {llvm.readnone}
// CHECK-DAG: @atan(f64) -> f64 attributes {llvm.readnone}
// CHECK-DAG: @atanf(f32) -> f32 attributes {llvm.readnone}
// CHECK-DAG: @erf(f64) -> f64 attributes {llvm.readnone}
// CHECK-DAG: @erff(f32) -> f32 attributes {llvm.readnone}
// CHECK-DAG: @expm1(f64) -> f64 attributes {llvm.readnone}
// CHECK-DAG: @expm1f(f32) -> f32 attributes {llvm.readnone}
// CHECK-DAG: @atan2(f64, f64) -> f64 attributes {llvm.readnone}
// CHECK-DAG: @atan2f(f32, f32) -> f32 attributes {llvm.readnone}
// CHECK-DAG: @cbrt(f64) -> f64 attributes {llvm.readnone}
// CHECK-DAG: @cbrtf(f32) -> f32 attributes {llvm.readnone}
// CHECK-DAG: @tan(f64) -> f64 attributes {llvm.readnone}
// CHECK-DAG: @tanf(f32) -> f32 attributes {llvm.readnone}
// CHECK-DAG: @tanh(f64) -> f64 attributes {llvm.readnone}
// CHECK-DAG: @tanhf(f32) -> f32 attributes {llvm.readnone}
// CHECK-DAG: @round(f64) -> f64 attributes {llvm.readnone}
// CHECK-DAG: @roundf(f32) -> f32 attributes {llvm.readnone}
// CHECK-DAG: @roundeven(f64) -> f64 attributes {llvm.readnone}
// CHECK-DAG: @roundevenf(f32) -> f32 attributes {llvm.readnone}
// CHECK-DAG: @trunc(f64) -> f64 attributes {llvm.readnone}
// CHECK-DAG: @truncf(f32) -> f32 attributes {llvm.readnone}
// CHECK-DAG: @cos(f64) -> f64 attributes {llvm.readnone}
// CHECK-DAG: @cosf(f32) -> f32 attributes {llvm.readnone}
// CHECK-DAG: @cosh(f64) -> f64 attributes {llvm.readnone}
// CHECK-DAG: @coshf(f32) -> f32 attributes {llvm.readnone}
// CHECK-DAG: @sin(f64) -> f64 attributes {llvm.readnone}
// CHECK-DAG: @sinf(f32) -> f32 attributes {llvm.readnone}
// CHECK-DAG: @floor(f64) -> f64 attributes {llvm.readnone}
// CHECK-DAG: @floorf(f32) -> f32 attributes {llvm.readnone}
// CHECK-DAG: @ceil(f64) -> f64 attributes {llvm.readnone}
// CHECK-DAG: @ceilf(f32) -> f32 attributes {llvm.readnone}

// CHECK-LABEL: func @acos_caller
// CHECK-SAME: %[[FLOAT:.*]]: f32
// CHECK-SAME: %[[DOUBLE:.*]]: f64
func.func @acos_caller(%float: f32, %double: f64) -> (f32, f64)  {
  // CHECK-DAG: %[[FLOAT_RESULT:.*]] = call @acosf(%[[FLOAT]]) : (f32) -> f32
  %float_result = math.acos %float : f32
  // CHECK-DAG: %[[DOUBLE_RESULT:.*]] = call @acos(%[[DOUBLE]]) : (f64) -> f64
  %double_result = math.acos %double : f64
  // CHECK: return %[[FLOAT_RESULT]], %[[DOUBLE_RESULT]]
  return %float_result, %double_result : f32, f64
}

// CHECK-LABEL:   func @acos_vec_caller(
// CHECK-SAME:                           %[[VAL_0:.*]]: vector<2xf32>,
// CHECK-SAME:                           %[[VAL_1:.*]]: vector<2xf64>) -> (vector<2xf32>, vector<2xf64>) {
// CHECK-DAG:       %[[CVF:.*]] = arith.constant dense<0.000000e+00> : vector<2xf32>
// CHECK-DAG:       %[[CVD:.*]] = arith.constant dense<0.000000e+00> : vector<2xf64>
// CHECK:           %[[IN0_F32:.*]] = vector.extract %[[VAL_0]][0] : f32 from vector<2xf32>
// CHECK:           %[[OUT0_F32:.*]] = call @acosf(%[[IN0_F32]]) : (f32) -> f32
// CHECK:           %[[VAL_8:.*]] = vector.insert %[[OUT0_F32]], %[[CVF]] [0] : f32 into vector<2xf32>
// CHECK:           %[[IN1_F32:.*]] = vector.extract %[[VAL_0]][1] : f32 from vector<2xf32>
// CHECK:           %[[OUT1_F32:.*]] = call @acosf(%[[IN1_F32]]) : (f32) -> f32
// CHECK:           %[[VAL_11:.*]] = vector.insert %[[OUT1_F32]], %[[VAL_8]] [1] : f32 into vector<2xf32>
// CHECK:           %[[IN0_F64:.*]] = vector.extract %[[VAL_1]][0] : f64 from vector<2xf64>
// CHECK:           %[[OUT0_F64:.*]] = call @acos(%[[IN0_F64]]) : (f64) -> f64
// CHECK:           %[[VAL_14:.*]] = vector.insert %[[OUT0_F64]], %[[CVD]] [0] : f64 into vector<2xf64>
// CHECK:           %[[IN1_F64:.*]] = vector.extract %[[VAL_1]][1] : f64 from vector<2xf64>
// CHECK:           %[[OUT1_F64:.*]] = call @acos(%[[IN1_F64]]) : (f64) -> f64
// CHECK:           %[[VAL_17:.*]] = vector.insert %[[OUT1_F64]], %[[VAL_14]] [1] : f64 into vector<2xf64>
// CHECK:           return %[[VAL_11]], %[[VAL_17]] : vector<2xf32>, vector<2xf64>
// CHECK:         }
func.func @acos_vec_caller(%float: vector<2xf32>, %double: vector<2xf64>) -> (vector<2xf32>, vector<2xf64>) {
  %float_result = math.acos %float : vector<2xf32>
  %double_result = math.acos %double : vector<2xf64>
  return %float_result, %double_result : vector<2xf32>, vector<2xf64>
}

// CHECK-LABEL: func @atan_caller
// CHECK-SAME: %[[FLOAT:.*]]: f32
// CHECK-SAME: %[[DOUBLE:.*]]: f64
// CHECK-SAME: %[[HALF:.*]]: f16
// CHECK-SAME: %[[BFLOAT:.*]]: bf16
func.func @atan_caller(%float: f32, %double: f64, %half: f16, %bfloat: bf16) -> (f32, f64, f16, bf16)  {
  // CHECK: %[[FLOAT_RESULT:.*]] = call @atanf(%[[FLOAT]]) : (f32) -> f32
  %float_result = math.atan %float : f32
  // CHECK: %[[DOUBLE_RESULT:.*]] = call @atan(%[[DOUBLE]]) : (f64) -> f64
  %double_result = math.atan %double : f64
  // CHECK: %[[HALF_PROMOTED:.*]] = arith.extf %[[HALF]] : f16 to f32
  // CHECK: %[[HALF_CALL:.*]] = call @atanf(%[[HALF_PROMOTED]]) : (f32) -> f32
  // CHECK: %[[HALF_RESULT:.*]] = arith.truncf %[[HALF_CALL]] : f32 to f16
  %half_result = math.atan %half : f16
  // CHECK: %[[BFLOAT_PROMOTED:.*]] = arith.extf %[[BFLOAT]] : bf16 to f32
  // CHECK: %[[BFLOAT_CALL:.*]] = call @atanf(%[[BFLOAT_PROMOTED]]) : (f32) -> f32
  // CHECK: %[[BFLOAT_RESULT:.*]] = arith.truncf %[[BFLOAT_CALL]] : f32 to bf16
  %bfloat_result = math.atan %bfloat : bf16
  // CHECK: return %[[FLOAT_RESULT]], %[[DOUBLE_RESULT]], %[[HALF_RESULT]], %[[BFLOAT_RESULT]]
  return %float_result, %double_result, %half_result, %bfloat_result : f32, f64, f16, bf16
}

// CHECK-LABEL:   func @atan_vec_caller(
// CHECK-SAME:                           %[[VAL_0:.*]]: vector<2xf32>,
// CHECK-SAME:                           %[[VAL_1:.*]]: vector<2xf64>) -> (vector<2xf32>, vector<2xf64>) {
// CHECK-DAG:       %[[CVF:.*]] = arith.constant dense<0.000000e+00> : vector<2xf32>
// CHECK-DAG:       %[[CVD:.*]] = arith.constant dense<0.000000e+00> : vector<2xf64>
// CHECK:           %[[IN0_F32:.*]] = vector.extract %[[VAL_0]][0] : f32 from vector<2xf32>
// CHECK:           %[[OUT0_F32:.*]] = call @atanf(%[[IN0_F32]]) : (f32) -> f32
// CHECK:           %[[VAL_8:.*]] = vector.insert %[[OUT0_F32]], %[[CVF]] [0] : f32 into vector<2xf32>
// CHECK:           %[[IN1_F32:.*]] = vector.extract %[[VAL_0]][1] : f32 from vector<2xf32>
// CHECK:           %[[OUT1_F32:.*]] = call @atanf(%[[IN1_F32]]) : (f32) -> f32
// CHECK:           %[[VAL_11:.*]] = vector.insert %[[OUT1_F32]], %[[VAL_8]] [1] : f32 into vector<2xf32>
// CHECK:           %[[IN0_F64:.*]] = vector.extract %[[VAL_1]][0] : f64 from vector<2xf64>
// CHECK:           %[[OUT0_F64:.*]] = call @atan(%[[IN0_F64]]) : (f64) -> f64
// CHECK:           %[[VAL_14:.*]] = vector.insert %[[OUT0_F64]], %[[CVD]] [0] : f64 into vector<2xf64>
// CHECK:           %[[IN1_F64:.*]] = vector.extract %[[VAL_1]][1] : f64 from vector<2xf64>
// CHECK:           %[[OUT1_F64:.*]] = call @atan(%[[IN1_F64]]) : (f64) -> f64
// CHECK:           %[[VAL_17:.*]] = vector.insert %[[OUT1_F64]], %[[VAL_14]] [1] : f64 into vector<2xf64>
// CHECK:           return %[[VAL_11]], %[[VAL_17]] : vector<2xf32>, vector<2xf64>
// CHECK:         }
func.func @atan_vec_caller(%float: vector<2xf32>, %double: vector<2xf64>) -> (vector<2xf32>, vector<2xf64>) {
  %float_result = math.atan %float : vector<2xf32>
  %double_result = math.atan %double : vector<2xf64>
  return %float_result, %double_result : vector<2xf32>, vector<2xf64>
}

// CHECK-LABEL: func @tanh_caller
// CHECK-SAME: %[[FLOAT:.*]]: f32
// CHECK-SAME: %[[DOUBLE:.*]]: f64
func.func @tanh_caller(%float: f32, %double: f64) -> (f32, f64)  {
  // CHECK-DAG: %[[FLOAT_RESULT:.*]] = call @tanhf(%[[FLOAT]]) : (f32) -> f32
  %float_result = math.tanh %float : f32
  // CHECK-DAG: %[[DOUBLE_RESULT:.*]] = call @tanh(%[[DOUBLE]]) : (f64) -> f64
  %double_result = math.tanh %double : f64
  // CHECK: return %[[FLOAT_RESULT]], %[[DOUBLE_RESULT]]
  return %float_result, %double_result : f32, f64
}

// CHECK-LABEL: func @cosh_caller
// CHECK-SAME: %[[FLOAT:.*]]: f32
// CHECK-SAME: %[[DOUBLE:.*]]: f64
func.func @cosh_caller(%float: f32, %double: f64) -> (f32, f64)  {
  // CHECK-DAG: %[[FLOAT_RESULT:.*]] = call @coshf(%[[FLOAT]]) : (f32) -> f32
  %float_result = math.cosh %float : f32
  // CHECK-DAG: %[[DOUBLE_RESULT:.*]] = call @cosh(%[[DOUBLE]]) : (f64) -> f64
  %double_result = math.cosh %double : f64
  // CHECK: return %[[FLOAT_RESULT]], %[[DOUBLE_RESULT]]
  return %float_result, %double_result : f32, f64
}

// CHECK-LABEL: func @sinh_caller
// CHECK-SAME: %[[FLOAT:.*]]: f32
// CHECK-SAME: %[[DOUBLE:.*]]: f64
func.func @sinh_caller(%float: f32, %double: f64) -> (f32, f64)  {
  // CHECK-DAG: %[[FLOAT_RESULT:.*]] = call @sinhf(%[[FLOAT]]) : (f32) -> f32
  %float_result = math.sinh %float : f32
  // CHECK-DAG: %[[DOUBLE_RESULT:.*]] = call @sinh(%[[DOUBLE]]) : (f64) -> f64
  %double_result = math.sinh %double : f64
  // CHECK: return %[[FLOAT_RESULT]], %[[DOUBLE_RESULT]]
  return %float_result, %double_result : f32, f64
}

// CHECK-LABEL: func @atan2_caller
// CHECK-SAME: %[[FLOAT:.*]]: f32
// CHECK-SAME: %[[DOUBLE:.*]]: f64
// CHECK-SAME: %[[HALF:.*]]: f16
// CHECK-SAME: %[[BFLOAT:.*]]: bf16
func.func @atan2_caller(%float: f32, %double: f64, %half: f16, %bfloat: bf16) -> (f32, f64, f16, bf16) {
  // CHECK: %[[FLOAT_RESULT:.*]] = call @atan2f(%[[FLOAT]], %[[FLOAT]]) : (f32, f32) -> f32
  %float_result = math.atan2 %float, %float : f32
  // CHECK: %[[DOUBLE_RESULT:.*]] = call @atan2(%[[DOUBLE]], %[[DOUBLE]]) : (f64, f64) -> f64
  %double_result = math.atan2 %double, %double : f64
  // CHECK: %[[HALF_PROMOTED1:.*]] = arith.extf %[[HALF]] : f16 to f32
  // CHECK: %[[HALF_PROMOTED2:.*]] = arith.extf %[[HALF]] : f16 to f32
  // CHECK: %[[HALF_CALL:.*]] = call @atan2f(%[[HALF_PROMOTED1]], %[[HALF_PROMOTED2]]) : (f32, f32) -> f32
  // CHECK: %[[HALF_RESULT:.*]] = arith.truncf %[[HALF_CALL]] : f32 to f16
  %half_result = math.atan2 %half, %half : f16
  // CHECK: %[[BFLOAT_PROMOTED1:.*]] = arith.extf %[[BFLOAT]] : bf16 to f32
  // CHECK: %[[BFLOAT_PROMOTED2:.*]] = arith.extf %[[BFLOAT]] : bf16 to f32
  // CHECK: %[[BFLOAT_CALL:.*]] = call @atan2f(%[[BFLOAT_PROMOTED1]], %[[BFLOAT_PROMOTED2]]) : (f32, f32) -> f32
  // CHECK: %[[BFLOAT_RESULT:.*]] = arith.truncf %[[BFLOAT_CALL]] : f32 to bf16
  %bfloat_result = math.atan2 %bfloat, %bfloat : bf16
  // CHECK: return %[[FLOAT_RESULT]], %[[DOUBLE_RESULT]], %[[HALF_RESULT]], %[[BFLOAT_RESULT]]
  return %float_result, %double_result, %half_result, %bfloat_result : f32, f64, f16, bf16
}

// CHECK-LABEL: func @erf_caller
// CHECK-SAME: %[[FLOAT:.*]]: f32
// CHECK-SAME: %[[DOUBLE:.*]]: f64
func.func @erf_caller(%float: f32, %double: f64) -> (f32, f64)  {
  // CHECK-DAG: %[[FLOAT_RESULT:.*]] = call @erff(%[[FLOAT]]) : (f32) -> f32
  %float_result = math.erf %float : f32
  // CHECK-DAG: %[[DOUBLE_RESULT:.*]] = call @erf(%[[DOUBLE]]) : (f64) -> f64
  %double_result = math.erf %double : f64
  // CHECK: return %[[FLOAT_RESULT]], %[[DOUBLE_RESULT]]
  return %float_result, %double_result : f32, f64
}

// CHECK-LABEL:   func @erf_vec_caller(
// CHECK-SAME:                           %[[VAL_0:.*]]: vector<2xf32>,
// CHECK-SAME:                           %[[VAL_1:.*]]: vector<2xf64>) -> (vector<2xf32>, vector<2xf64>) {
func.func @erf_vec_caller(%float: vector<2xf32>, %double: vector<2xf64>) -> (vector<2xf32>, vector<2xf64>) {
  // CHECK-DAG:       %[[CVF:.*]] = arith.constant dense<0.000000e+00> : vector<2xf32>
  // CHECK-DAG:       %[[CVD:.*]] = arith.constant dense<0.000000e+00> : vector<2xf64>
  // CHECK:           %[[IN0_F32:.*]] = vector.extract %[[VAL_0]][0] : f32 from vector<2xf32>
  // CHECK:           %[[OUT0_F32:.*]] = call @erff(%[[IN0_F32]]) : (f32) -> f32
  // CHECK:           %[[VAL_8:.*]] = vector.insert %[[OUT0_F32]], %[[CVF]] [0] : f32 into vector<2xf32>
  // CHECK:           %[[IN1_F32:.*]] = vector.extract %[[VAL_0]][1] : f32 from vector<2xf32>
  // CHECK:           %[[OUT1_F32:.*]] = call @erff(%[[IN1_F32]]) : (f32) -> f32
  // CHECK:           %[[VAL_11:.*]] = vector.insert %[[OUT1_F32]], %[[VAL_8]] [1] : f32 into vector<2xf32>
  %float_result = math.erf %float : vector<2xf32>
  // CHECK:           %[[IN0_F64:.*]] = vector.extract %[[VAL_1]][0] : f64 from vector<2xf64>
  // CHECK:           %[[OUT0_F64:.*]] = call @erf(%[[IN0_F64]]) : (f64) -> f64
  // CHECK:           %[[VAL_14:.*]] = vector.insert %[[OUT0_F64]], %[[CVD]] [0] : f64 into vector<2xf64>
  // CHECK:           %[[IN1_F64:.*]] = vector.extract %[[VAL_1]][1] : f64 from vector<2xf64>
  // CHECK:           %[[OUT1_F64:.*]] = call @erf(%[[IN1_F64]]) : (f64) -> f64
  // CHECK:           %[[VAL_17:.*]] = vector.insert %[[OUT1_F64]], %[[VAL_14]] [1] : f64 into vector<2xf64>
  %double_result = math.erf %double : vector<2xf64>
  // CHECK:           return %[[VAL_11]], %[[VAL_17]] : vector<2xf32>, vector<2xf64>
  return %float_result, %double_result : vector<2xf32>, vector<2xf64>
}

// CHECK-LABEL: func @expm1_caller
// CHECK-SAME: %[[FLOAT:.*]]: f32
// CHECK-SAME: %[[DOUBLE:.*]]: f64
func.func @expm1_caller(%float: f32, %double: f64) -> (f32, f64) {
  // CHECK-DAG: %[[FLOAT_RESULT:.*]] = call @expm1f(%[[FLOAT]]) : (f32) -> f32
  %float_result = math.expm1 %float : f32
  // CHECK-DAG: %[[DOUBLE_RESULT:.*]] = call @expm1(%[[DOUBLE]]) : (f64) -> f64
  %double_result = math.expm1 %double : f64
  // CHECK: return %[[FLOAT_RESULT]], %[[DOUBLE_RESULT]]
  return %float_result, %double_result : f32, f64
}

func.func @expm1_vec_caller(%float: vector<2xf32>, %double: vector<2xf64>) -> (vector<2xf32>, vector<2xf64>) {
  %float_result = math.expm1 %float : vector<2xf32>
  %double_result = math.expm1 %double : vector<2xf64>
  return %float_result, %double_result : vector<2xf32>, vector<2xf64>
}
// CHECK-LABEL:   func @expm1_vec_caller(
// CHECK-SAME:                           %[[VAL_0:.*]]: vector<2xf32>,
// CHECK-SAME:                           %[[VAL_1:.*]]: vector<2xf64>) -> (vector<2xf32>, vector<2xf64>) {
// CHECK-DAG:       %[[CVF:.*]] = arith.constant dense<0.000000e+00> : vector<2xf32>
// CHECK-DAG:       %[[CVD:.*]] = arith.constant dense<0.000000e+00> : vector<2xf64>
// CHECK:           %[[IN0_F32:.*]] = vector.extract %[[VAL_0]][0] : f32 from vector<2xf32>
// CHECK:           %[[OUT0_F32:.*]] = call @expm1f(%[[IN0_F32]]) : (f32) -> f32
// CHECK:           %[[VAL_8:.*]] = vector.insert %[[OUT0_F32]], %[[CVF]] [0] : f32 into vector<2xf32>
// CHECK:           %[[IN1_F32:.*]] = vector.extract %[[VAL_0]][1] : f32 from vector<2xf32>
// CHECK:           %[[OUT1_F32:.*]] = call @expm1f(%[[IN1_F32]]) : (f32) -> f32
// CHECK:           %[[VAL_11:.*]] = vector.insert %[[OUT1_F32]], %[[VAL_8]] [1] : f32 into vector<2xf32>
// CHECK:           %[[IN0_F64:.*]] = vector.extract %[[VAL_1]][0] : f64 from vector<2xf64>
// CHECK:           %[[OUT0_F64:.*]] = call @expm1(%[[IN0_F64]]) : (f64) -> f64
// CHECK:           %[[VAL_14:.*]] = vector.insert %[[OUT0_F64]], %[[CVD]] [0] : f64 into vector<2xf64>
// CHECK:           %[[IN1_F64:.*]] = vector.extract %[[VAL_1]][1] : f64 from vector<2xf64>
// CHECK:           %[[OUT1_F64:.*]] = call @expm1(%[[IN1_F64]]) : (f64) -> f64
// CHECK:           %[[VAL_17:.*]] = vector.insert %[[OUT1_F64]], %[[VAL_14]] [1] : f64 into vector<2xf64>
// CHECK:           return %[[VAL_11]], %[[VAL_17]] : vector<2xf32>, vector<2xf64>
// CHECK:         }

func.func @expm1_multidim_vec_caller(%float: vector<2x2xf32>) -> (vector<2x2xf32>) {
  %result = math.expm1 %float : vector<2x2xf32>
  return %result : vector<2x2xf32>
}
// CHECK-LABEL:   func @expm1_multidim_vec_caller(
// CHECK-SAME:                           %[[VAL:.*]]: vector<2x2xf32>
// CHECK-DAG:       %[[CVF:.*]] = arith.constant dense<0.000000e+00> : vector<2x2xf32>
// CHECK:           %[[IN0_0_F32:.*]] = vector.extract %[[VAL]][0, 0] : f32 from vector<2x2xf32>
// CHECK:           %[[OUT0_0_F32:.*]] = call @expm1f(%[[IN0_0_F32]]) : (f32) -> f32
// CHECK:           %[[VAL_1:.*]] = vector.insert %[[OUT0_0_F32]], %[[CVF]] [0, 0] : f32 into vector<2x2xf32>
// CHECK:           %[[IN0_1_F32:.*]] = vector.extract %[[VAL]][0, 1] : f32 from vector<2x2xf32>
// CHECK:           %[[OUT0_1_F32:.*]] = call @expm1f(%[[IN0_1_F32]]) : (f32) -> f32
// CHECK:           %[[VAL_2:.*]] = vector.insert %[[OUT0_1_F32]], %[[VAL_1]] [0, 1] : f32 into vector<2x2xf32>
// CHECK:           %[[IN1_0_F32:.*]] = vector.extract %[[VAL]][1, 0] : f32 from vector<2x2xf32>
// CHECK:           %[[OUT1_0_F32:.*]] = call @expm1f(%[[IN1_0_F32]]) : (f32) -> f32
// CHECK:           %[[VAL_3:.*]] = vector.insert %[[OUT1_0_F32]], %[[VAL_2]] [1, 0] : f32 into vector<2x2xf32>
// CHECK:           %[[IN1_1_F32:.*]] = vector.extract %[[VAL]][1, 1] : f32 from vector<2x2xf32>
// CHECK:           %[[OUT1_1_F32:.*]] = call @expm1f(%[[IN1_1_F32]]) : (f32) -> f32
// CHECK:           %[[VAL_4:.*]] = vector.insert %[[OUT1_1_F32]], %[[VAL_3]] [1, 1] : f32 into vector<2x2xf32>
// CHECK:           return %[[VAL_4]] : vector<2x2xf32>
// CHECK:         }

// CHECK-LABEL: func @round_caller
// CHECK-SAME: %[[FLOAT:.*]]: f32
// CHECK-SAME: %[[DOUBLE:.*]]: f64
func.func @round_caller(%float: f32, %double: f64) -> (f32, f64) {
  // CHECK-DAG: %[[FLOAT_RESULT:.*]] = call @roundf(%[[FLOAT]]) : (f32) -> f32
  %float_result = math.round %float : f32
  // CHECK-DAG: %[[DOUBLE_RESULT:.*]] = call @round(%[[DOUBLE]]) : (f64) -> f64
  %double_result = math.round %double : f64
  // CHECK: return %[[FLOAT_RESULT]], %[[DOUBLE_RESULT]]
  return %float_result, %double_result : f32, f64
}

// CHECK-LABEL: func @roundeven_caller
// CHECK-SAME: %[[FLOAT:.*]]: f32
// CHECK-SAME: %[[DOUBLE:.*]]: f64
func.func @roundeven_caller(%float: f32, %double: f64) -> (f32, f64) {
  // CHECK-DAG: %[[FLOAT_RESULT:.*]] = call @roundevenf(%[[FLOAT]]) : (f32) -> f32
  %float_result = math.roundeven %float : f32
  // CHECK-DAG: %[[DOUBLE_RESULT:.*]] = call @roundeven(%[[DOUBLE]]) : (f64) -> f64
  %double_result = math.roundeven %double : f64
  // CHECK: return %[[FLOAT_RESULT]], %[[DOUBLE_RESULT]]
  return %float_result, %double_result : f32, f64
}

// CHECK-LABEL: func @trunc_caller
// CHECK-SAME: %[[FLOAT:.*]]: f32
// CHECK-SAME: %[[DOUBLE:.*]]: f64
func.func @trunc_caller(%float: f32, %double: f64) -> (f32, f64) {
  // CHECK-DAG: %[[FLOAT_RESULT:.*]] = call @truncf(%[[FLOAT]]) : (f32) -> f32
  %float_result = math.trunc %float : f32
  // CHECK-DAG: %[[DOUBLE_RESULT:.*]] = call @trunc(%[[DOUBLE]]) : (f64) -> f64
  %double_result = math.trunc %double : f64
  // CHECK: return %[[FLOAT_RESULT]], %[[DOUBLE_RESULT]]
  return %float_result, %double_result : f32, f64
}

// CHECK-LABEL: func @cbrt_caller
// CHECK-SAME: %[[FLOAT:.*]]: f32
// CHECK-SAME: %[[DOUBLE:.*]]: f64
func.func @cbrt_caller(%float: f32, %double: f64, %half: f16, %bfloat: bf16,
                       %float_vec: vector<2xf32>) -> (f32, f64, f16, bf16, vector<2xf32>)  {
  // CHECK: %[[FLOAT_RESULT:.*]] = call @cbrtf(%[[FLOAT]]) : (f32) -> f32
  %float_result = math.cbrt %float : f32
  // CHECK: %[[DOUBLE_RESULT:.*]] = call @cbrt(%[[DOUBLE]]) : (f64) -> f64
  %double_result = math.cbrt %double : f64
  // Just check that these lower successfully:
  // CHECK: call @cbrtf
  %half_result = math.cbrt %half : f16
  // CHECK: call @cbrtf
  %bfloat_result = math.cbrt %bfloat : bf16
  // CHECK: call @cbrtf
  %vec_result = math.cbrt %float_vec : vector<2xf32>
  // CHECK: return %[[FLOAT_RESULT]], %[[DOUBLE_RESULT]]
  return %float_result, %double_result, %half_result, %bfloat_result, %vec_result
    : f32, f64, f16, bf16, vector<2xf32>
}

// CHECK-LABEL: func @cos_caller
// CHECK-SAME: %[[FLOAT:.*]]: f32
// CHECK-SAME: %[[DOUBLE:.*]]: f64
func.func @cos_caller(%float: f32, %double: f64) -> (f32, f64)  {
  // CHECK-DAG: %[[FLOAT_RESULT:.*]] = call @cosf(%[[FLOAT]]) : (f32) -> f32
  %float_result = math.cos %float : f32
  // CHECK-DAG: %[[DOUBLE_RESULT:.*]] = call @cos(%[[DOUBLE]]) : (f64) -> f64
  %double_result = math.cos %double : f64
  // CHECK: return %[[FLOAT_RESULT]], %[[DOUBLE_RESULT]]
  return %float_result, %double_result : f32, f64
}

// CHECK-LABEL: func @sin_caller
// CHECK-SAME: %[[FLOAT:.*]]: f32
// CHECK-SAME: %[[DOUBLE:.*]]: f64
func.func @sin_caller(%float: f32, %double: f64) -> (f32, f64)  {
  // CHECK-DAG: %[[FLOAT_RESULT:.*]] = call @sinf(%[[FLOAT]]) : (f32) -> f32
  %float_result = math.sin %float : f32
  // CHECK-DAG: %[[DOUBLE_RESULT:.*]] = call @sin(%[[DOUBLE]]) : (f64) -> f64
  %double_result = math.sin %double : f64
  // CHECK: return %[[FLOAT_RESULT]], %[[DOUBLE_RESULT]]
  return %float_result, %double_result : f32, f64
}

// CHECK-LABEL:   func @round_vec_caller(
// CHECK-SAME:                           %[[VAL_0:.*]]: vector<2xf32>,
// CHECK-SAME:                           %[[VAL_1:.*]]: vector<2xf64>) -> (vector<2xf32>, vector<2xf64>) {
func.func @round_vec_caller(%float: vector<2xf32>, %double: vector<2xf64>) -> (vector<2xf32>, vector<2xf64>) {
  // CHECK-DAG:       %[[CVF:.*]] = arith.constant dense<0.000000e+00> : vector<2xf32>
  // CHECK-DAG:       %[[CVD:.*]] = arith.constant dense<0.000000e+00> : vector<2xf64>
  // CHECK:           %[[IN0_F32:.*]] = vector.extract %[[VAL_0]][0] : f32 from vector<2xf32>
  // CHECK:           %[[OUT0_F32:.*]] = call @roundf(%[[IN0_F32]]) : (f32) -> f32
  // CHECK:           %[[VAL_8:.*]] = vector.insert %[[OUT0_F32]], %[[CVF]] [0] : f32 into vector<2xf32>
  // CHECK:           %[[IN1_F32:.*]] = vector.extract %[[VAL_0]][1] : f32 from vector<2xf32>
  // CHECK:           %[[OUT1_F32:.*]] = call @roundf(%[[IN1_F32]]) : (f32) -> f32
  // CHECK:           %[[VAL_11:.*]] = vector.insert %[[OUT1_F32]], %[[VAL_8]] [1] : f32 into vector<2xf32>
  %float_result = math.round %float : vector<2xf32>
  // CHECK:           %[[IN0_F64:.*]] = vector.extract %[[VAL_1]][0] : f64 from vector<2xf64>
  // CHECK:           %[[OUT0_F64:.*]] = call @round(%[[IN0_F64]]) : (f64) -> f64
  // CHECK:           %[[VAL_14:.*]] = vector.insert %[[OUT0_F64]], %[[CVD]] [0] : f64 into vector<2xf64>
  // CHECK:           %[[IN1_F64:.*]] = vector.extract %[[VAL_1]][1] : f64 from vector<2xf64>
  // CHECK:           %[[OUT1_F64:.*]] = call @round(%[[IN1_F64]]) : (f64) -> f64
  // CHECK:           %[[VAL_17:.*]] = vector.insert %[[OUT1_F64]], %[[VAL_14]] [1] : f64 into vector<2xf64>
  %double_result = math.round %double : vector<2xf64>
  // CHECK:           return %[[VAL_11]], %[[VAL_17]] : vector<2xf32>, vector<2xf64>
  return %float_result, %double_result : vector<2xf32>, vector<2xf64>
}

// CHECK-LABEL:   func @roundeven_vec_caller(
// CHECK-SAME:                           %[[VAL_0:.*]]: vector<2xf32>,
// CHECK-SAME:                           %[[VAL_1:.*]]: vector<2xf64>) -> (vector<2xf32>, vector<2xf64>) {
func.func @roundeven_vec_caller(%float: vector<2xf32>, %double: vector<2xf64>) -> (vector<2xf32>, vector<2xf64>) {
  // CHECK-DAG:       %[[CVF:.*]] = arith.constant dense<0.000000e+00> : vector<2xf32>
  // CHECK-DAG:       %[[CVD:.*]] = arith.constant dense<0.000000e+00> : vector<2xf64>
  // CHECK:           %[[IN0_F32:.*]] = vector.extract %[[VAL_0]][0] : f32 from vector<2xf32>
  // CHECK:           %[[OUT0_F32:.*]] = call @roundevenf(%[[IN0_F32]]) : (f32) -> f32
  // CHECK:           %[[VAL_8:.*]] = vector.insert %[[OUT0_F32]], %[[CVF]] [0] : f32 into vector<2xf32>
  // CHECK:           %[[IN1_F32:.*]] = vector.extract %[[VAL_0]][1] : f32 from vector<2xf32>
  // CHECK:           %[[OUT1_F32:.*]] = call @roundevenf(%[[IN1_F32]]) : (f32) -> f32
  // CHECK:           %[[VAL_11:.*]] = vector.insert %[[OUT1_F32]], %[[VAL_8]] [1] : f32 into vector<2xf32>
  %float_result = math.roundeven %float : vector<2xf32>
  // CHECK:           %[[IN0_F64:.*]] = vector.extract %[[VAL_1]][0] : f64 from vector<2xf64>
  // CHECK:           %[[OUT0_F64:.*]] = call @roundeven(%[[IN0_F64]]) : (f64) -> f64
  // CHECK:           %[[VAL_14:.*]] = vector.insert %[[OUT0_F64]], %[[CVD]] [0] : f64 into vector<2xf64>
  // CHECK:           %[[IN1_F64:.*]] = vector.extract %[[VAL_1]][1] : f64 from vector<2xf64>
  // CHECK:           %[[OUT1_F64:.*]] = call @roundeven(%[[IN1_F64]]) : (f64) -> f64
  // CHECK:           %[[VAL_17:.*]] = vector.insert %[[OUT1_F64]], %[[VAL_14]] [1] : f64 into vector<2xf64>
  %double_result = math.roundeven %double : vector<2xf64>
  // CHECK:           return %[[VAL_11]], %[[VAL_17]] : vector<2xf32>, vector<2xf64>
  return %float_result, %double_result : vector<2xf32>, vector<2xf64>
}

// CHECK-LABEL:   func @trunc_vec_caller(
// CHECK-SAME:                           %[[VAL_0:.*]]: vector<2xf32>,
// CHECK-SAME:                           %[[VAL_1:.*]]: vector<2xf64>) -> (vector<2xf32>, vector<2xf64>) {
func.func @trunc_vec_caller(%float: vector<2xf32>, %double: vector<2xf64>) -> (vector<2xf32>, vector<2xf64>) {
  // CHECK-DAG:       %[[CVF:.*]] = arith.constant dense<0.000000e+00> : vector<2xf32>
  // CHECK-DAG:       %[[CVD:.*]] = arith.constant dense<0.000000e+00> : vector<2xf64>
  // CHECK:           %[[IN0_F32:.*]] = vector.extract %[[VAL_0]][0] : f32 from vector<2xf32>
  // CHECK:           %[[OUT0_F32:.*]] = call @truncf(%[[IN0_F32]]) : (f32) -> f32
  // CHECK:           %[[VAL_8:.*]] = vector.insert %[[OUT0_F32]], %[[CVF]] [0] : f32 into vector<2xf32>
  // CHECK:           %[[IN1_F32:.*]] = vector.extract %[[VAL_0]][1] : f32 from vector<2xf32>
  // CHECK:           %[[OUT1_F32:.*]] = call @truncf(%[[IN1_F32]]) : (f32) -> f32
  // CHECK:           %[[VAL_11:.*]] = vector.insert %[[OUT1_F32]], %[[VAL_8]] [1] : f32 into vector<2xf32>
  %float_result = math.trunc %float : vector<2xf32>
  // CHECK:           %[[IN0_F64:.*]] = vector.extract %[[VAL_1]][0] : f64 from vector<2xf64>
  // CHECK:           %[[OUT0_F64:.*]] = call @trunc(%[[IN0_F64]]) : (f64) -> f64
  // CHECK:           %[[VAL_14:.*]] = vector.insert %[[OUT0_F64]], %[[CVD]] [0] : f64 into vector<2xf64>
  // CHECK:           %[[IN1_F64:.*]] = vector.extract %[[VAL_1]][1] : f64 from vector<2xf64>
  // CHECK:           %[[OUT1_F64:.*]] = call @trunc(%[[IN1_F64]]) : (f64) -> f64
  // CHECK:           %[[VAL_17:.*]] = vector.insert %[[OUT1_F64]], %[[VAL_14]] [1] : f64 into vector<2xf64>
  %double_result = math.trunc %double : vector<2xf64>
  // CHECK:           return %[[VAL_11]], %[[VAL_17]] : vector<2xf32>, vector<2xf64>
  return %float_result, %double_result : vector<2xf32>, vector<2xf64>
}

// CHECK-LABEL: func @tan_caller
// CHECK-SAME: %[[FLOAT:.*]]: f32
// CHECK-SAME: %[[DOUBLE:.*]]: f64
// CHECK-SAME: %[[HALF:.*]]: f16
// CHECK-SAME: %[[BFLOAT:.*]]: bf16
func.func @tan_caller(%float: f32, %double: f64, %half: f16, %bfloat: bf16) -> (f32, f64, f16, bf16)  {
  // CHECK: %[[FLOAT_RESULT:.*]] = call @tanf(%[[FLOAT]]) : (f32) -> f32
  %float_result = math.tan %float : f32
  // CHECK: %[[DOUBLE_RESULT:.*]] = call @tan(%[[DOUBLE]]) : (f64) -> f64
  %double_result = math.tan %double : f64
  // CHECK: %[[HALF_PROMOTED:.*]] = arith.extf %[[HALF]] : f16 to f32
  // CHECK: %[[HALF_CALL:.*]] = call @tanf(%[[HALF_PROMOTED]]) : (f32) -> f32
  // CHECK: %[[HALF_RESULT:.*]] = arith.truncf %[[HALF_CALL]] : f32 to f16
  %half_result = math.tan %half : f16
  // CHECK: %[[BFLOAT_PROMOTED:.*]] = arith.extf %[[BFLOAT]] : bf16 to f32
  // CHECK: %[[BFLOAT_CALL:.*]] = call @tanf(%[[BFLOAT_PROMOTED]]) : (f32) -> f32
  // CHECK: %[[BFLOAT_RESULT:.*]] = arith.truncf %[[BFLOAT_CALL]] : f32 to bf16
  %bfloat_result = math.tan %bfloat : bf16
  // CHECK: return %[[FLOAT_RESULT]], %[[DOUBLE_RESULT]], %[[HALF_RESULT]], %[[BFLOAT_RESULT]]
  return %float_result, %double_result, %half_result, %bfloat_result : f32, f64, f16, bf16
}

// CHECK-LABEL:   func @tan_vec_caller(
// CHECK-SAME:                           %[[VAL_0:.*]]: vector<2xf32>,
// CHECK-SAME:                           %[[VAL_1:.*]]: vector<2xf64>) -> (vector<2xf32>, vector<2xf64>) {
// CHECK-DAG:       %[[CVF:.*]] = arith.constant dense<0.000000e+00> : vector<2xf32>
// CHECK-DAG:       %[[CVD:.*]] = arith.constant dense<0.000000e+00> : vector<2xf64>
// CHECK:           %[[IN0_F32:.*]] = vector.extract %[[VAL_0]][0] : f32 from vector<2xf32>
// CHECK:           %[[OUT0_F32:.*]] = call @tanf(%[[IN0_F32]]) : (f32) -> f32
// CHECK:           %[[VAL_8:.*]] = vector.insert %[[OUT0_F32]], %[[CVF]] [0] : f32 into vector<2xf32>
// CHECK:           %[[IN1_F32:.*]] = vector.extract %[[VAL_0]][1] : f32 from vector<2xf32>
// CHECK:           %[[OUT1_F32:.*]] = call @tanf(%[[IN1_F32]]) : (f32) -> f32
// CHECK:           %[[VAL_11:.*]] = vector.insert %[[OUT1_F32]], %[[VAL_8]] [1] : f32 into vector<2xf32>
// CHECK:           %[[IN0_F64:.*]] = vector.extract %[[VAL_1]][0] : f64 from vector<2xf64>
// CHECK:           %[[OUT0_F64:.*]] = call @tan(%[[IN0_F64]]) : (f64) -> f64
// CHECK:           %[[VAL_14:.*]] = vector.insert %[[OUT0_F64]], %[[CVD]] [0] : f64 into vector<2xf64>
// CHECK:           %[[IN1_F64:.*]] = vector.extract %[[VAL_1]][1] : f64 from vector<2xf64>
// CHECK:           %[[OUT1_F64:.*]] = call @tan(%[[IN1_F64]]) : (f64) -> f64
// CHECK:           %[[VAL_17:.*]] = vector.insert %[[OUT1_F64]], %[[VAL_14]] [1] : f64 into vector<2xf64>
// CHECK:           return %[[VAL_11]], %[[VAL_17]] : vector<2xf32>, vector<2xf64>
// CHECK:         }
func.func @tan_vec_caller(%float: vector<2xf32>, %double: vector<2xf64>) -> (vector<2xf32>, vector<2xf64>) {
  %float_result = math.tan %float : vector<2xf32>
  %double_result = math.tan %double : vector<2xf64>
  return %float_result, %double_result : vector<2xf32>, vector<2xf64>
}

// CHECK-LABEL: func @log1p_caller
// CHECK-SAME: %[[FLOAT:.*]]: f32
// CHECK-SAME: %[[DOUBLE:.*]]: f64
func.func @log1p_caller(%float: f32, %double: f64) -> (f32, f64)  {
  // CHECK-DAG: %[[FLOAT_RESULT:.*]] = call @log1pf(%[[FLOAT]]) : (f32) -> f32
  %float_result = math.log1p %float : f32
  // CHECK-DAG: %[[DOUBLE_RESULT:.*]] = call @log1p(%[[DOUBLE]]) : (f64) -> f64
  %double_result = math.log1p %double : f64
  // CHECK: return %[[FLOAT_RESULT]], %[[DOUBLE_RESULT]]
  return %float_result, %double_result : f32, f64
}

// CHECK-LABEL: func @floor_caller
// CHECK-SAME: %[[FLOAT:.*]]: f32
// CHECK-SAME: %[[DOUBLE:.*]]: f64
func.func @floor_caller(%float: f32, %double: f64) -> (f32, f64)  {
  // CHECK-DAG: %[[FLOAT_RESULT:.*]] = call @floorf(%[[FLOAT]]) : (f32) -> f32
  %float_result = math.floor %float : f32
  // CHECK-DAG: %[[DOUBLE_RESULT:.*]] = call @floor(%[[DOUBLE]]) : (f64) -> f64
  %double_result = math.floor %double : f64
  // CHECK: return %[[FLOAT_RESULT]], %[[DOUBLE_RESULT]]
  return %float_result, %double_result : f32, f64
}

// CHECK-LABEL: func @ceil_caller
// CHECK-SAME: %[[FLOAT:.*]]: f32
// CHECK-SAME: %[[DOUBLE:.*]]: f64
func.func @ceil_caller(%float: f32, %double: f64) -> (f32, f64)  {
  // CHECK-DAG: %[[FLOAT_RESULT:.*]] = call @ceilf(%[[FLOAT]]) : (f32) -> f32
  %float_result = math.ceil %float : f32
  // CHECK-DAG: %[[DOUBLE_RESULT:.*]] = call @ceil(%[[DOUBLE]]) : (f64) -> f64
  %double_result = math.ceil %double : f64
  // CHECK: return %[[FLOAT_RESULT]], %[[DOUBLE_RESULT]]
  return %float_result, %double_result : f32, f64
}

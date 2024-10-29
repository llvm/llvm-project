// RUN: mlir-opt -convert-math-to-emitc %s | FileCheck %s


// CHECK-LABEL:   func.func @absf_to_call_opaque(
// CHECK-SAME:                                   %[[VAL_0:.*]]: f32) {
// CHECK:           %[[VAL_1:.*]] = emitc.call_opaque "fabsf"(%[[VAL_0]]) : (f32) -> f32
// CHECK:           return
// CHECK:         }
func.func @absf_to_call_opaque(%arg0: f32) {
    %1 = math.absf %arg0 : f32
    return
  }
// CHECK-LABEL:   func.func @floor_to_call_opaque(
// CHECK-SAME:                                    %[[VAL_0:.*]]: f32) {
// CHECK:           %[[VAL_1:.*]] = emitc.call_opaque "floorf"(%[[VAL_0]]) : (f32) -> f32
// CHECK:           return
// CHECK:         }
func.func @floor_to_call_opaque(%arg0: f32) {
    %1 = math.floor %arg0 : f32
    return
  }
// CHECK-LABEL:   func.func @sin_to_call_opaque(
// CHECK-SAME:                                  %[[VAL_0:.*]]: f32) {
// CHECK:           %[[VAL_1:.*]] = emitc.call_opaque "sinf"(%[[VAL_0]]) : (f32) -> f32
// CHECK:           return
// CHECK:         }
func.func @sin_to_call_opaque(%arg0: f32) {
    %1 = math.sin %arg0 : f32
    return
  }

// CHECK-LABEL:   func.func @cos_to_call_opaque(
// CHECK-SAME:                                  %[[VAL_0:.*]]: f32) {
// CHECK:           %[[VAL_1:.*]] = emitc.call_opaque "cosf"(%[[VAL_0]]) : (f32) -> f32
// CHECK:           return
// CHECK:         }
func.func @cos_to_call_opaque(%arg0: f32) {
    %1 = math.cos %arg0 : f32
    return
  }

// CHECK-LABEL:   func.func @asin_to_call_opaque(
// CHECK-SAME:                                   %[[VAL_0:.*]]: f32) {
// CHECK:           %[[VAL_1:.*]] = emitc.call_opaque "asinf"(%[[VAL_0]]) : (f32) -> f32
// CHECK:           return
// CHECK:         }
func.func @asin_to_call_opaque(%arg0: f32) {
    %1 = math.asin %arg0 : f32
    return
  }

// CHECK-LABEL:   func.func @acos_to_call_opaque(
// CHECK-SAME:                                   %[[VAL_0:.*]]: f32) {
// CHECK:           %[[VAL_1:.*]] = emitc.call_opaque "acosf"(%[[VAL_0]]) : (f32) -> f32
// CHECK:           return
// CHECK:         }
func.func @acos_to_call_opaque(%arg0: f32) {
    %1 = math.acos %arg0 : f32
    return
  }

// CHECK-LABEL:   func.func @atan2_to_call_opaque(
// CHECK-SAME:                                    %[[VAL_0:.*]]: f32,
// CHECK-SAME:                                    %[[VAL_1:.*]]: f32) {
// CHECK:           %[[VAL_2:.*]] = emitc.call_opaque "atan2f"(%[[VAL_0]], %[[VAL_1]]) : (f32, f32) -> f32
// CHECK:           return
// CHECK:         }
func.func @atan2_to_call_opaque(%arg0: f32, %arg1: f32) {
    %1 = math.atan2 %arg0, %arg1 : f32
    return
  }


// CHECK-LABEL:   func.func @ceil_to_call_opaque(
// CHECK-SAME:                                   %[[VAL_0:.*]]: f32) {
// CHECK:           %[[VAL_1:.*]] = emitc.call_opaque "ceilf"(%[[VAL_0]]) : (f32) -> f32
// CHECK:           return
// CHECK:         }
func.func @ceil_to_call_opaque(%arg0: f32) {
    %1 = math.ceil %arg0 : f32
    return
  }

// CHECK-LABEL:   func.func @exp_to_call_opaque(
// CHECK-SAME:                                  %[[VAL_0:.*]]: f32) {
// CHECK:           %[[VAL_1:.*]] = emitc.call_opaque "expf"(%[[VAL_0]]) : (f32) -> f32
// CHECK:           return
// CHECK:         }
func.func @exp_to_call_opaque(%arg0: f32) {
    %1 = math.exp %arg0 : f32
    return
  }


// CHECK-LABEL:   func.func @powf_to_call_opaque(
// CHECK-SAME:                                   %[[VAL_0:.*]]: f32,
// CHECK-SAME:                                   %[[VAL_1:.*]]: f32) {
// CHECK:           %[[VAL_2:.*]] = emitc.call_opaque "powf"(%[[VAL_0]], %[[VAL_1]]) : (f32, f32) -> f32
// CHECK:           return
// CHECK:         }
func.func @powf_to_call_opaque(%arg0: f32, %arg1: f32) {
    %1 = math.powf %arg0, %arg1 : f32
    return
  }



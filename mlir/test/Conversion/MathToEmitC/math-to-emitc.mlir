// RUN: mlir-opt --split-input-file -convert-math-to-emitc %s | FileCheck %s

// CHECK-LABEL:   emitc.include "math.h"

// CHECK-LABEL:   func.func @absf_to_call_opaque(
// CHECK-SAME:                                   %[[VAL_0:.*]]: f32) {
// CHECK:           %[[VAL_1:.*]] = emitc.call_opaque "fabs"(%[[VAL_0]]) : (f32) -> f32
// CHECK:           return
// CHECK:         }
func.func @absf_to_call_opaque(%arg0: f32) {
    %1 = math.absf %arg0 : f32
    return
  }

// -----

// CHECK-LABEL:   func.func @floor_to_call_opaque(
// CHECK-SAME:                                    %[[VAL_0:.*]]: f32) {
// CHECK:           %[[VAL_1:.*]] = emitc.call_opaque "floor"(%[[VAL_0]]) : (f32) -> f32
// CHECK:           return
// CHECK:         }
func.func @floor_to_call_opaque(%arg0: f32) {
    %1 = math.floor %arg0 : f32
    return
  }

// -----

// CHECK-LABEL:   func.func @sin_to_call_opaque(
// CHECK-SAME:                                  %[[VAL_0:.*]]: f32) {
// CHECK:           %[[VAL_1:.*]] = emitc.call_opaque "sin"(%[[VAL_0]]) : (f32) -> f32
// CHECK:           return
// CHECK:         }
func.func @sin_to_call_opaque(%arg0: f32) {
    %1 = math.sin %arg0 : f32
    return
  }

// -----

// CHECK-LABEL:   func.func @cos_to_call_opaque(
// CHECK-SAME:                                  %[[VAL_0:.*]]: f32) {
// CHECK:           %[[VAL_1:.*]] = emitc.call_opaque "cos"(%[[VAL_0]]) : (f32) -> f32
// CHECK:           return
// CHECK:         }
func.func @cos_to_call_opaque(%arg0: f32) {
    %1 = math.cos %arg0 : f32
    return
  }


// -----

// CHECK-LABEL:   func.func @asin_to_call_opaque(
// CHECK-SAME:                                   %[[VAL_0:.*]]: f32) {
// CHECK:           %[[VAL_1:.*]] = emitc.call_opaque "asin"(%[[VAL_0]]) : (f32) -> f32
// CHECK:           return
// CHECK:         }
func.func @asin_to_call_opaque(%arg0: f32) {
    %1 = math.asin %arg0 : f32
    return
  }

// -----

// CHECK-LABEL:   func.func @acos_to_call_opaque(
// CHECK-SAME:                                   %[[VAL_0:.*]]: f32) {
// CHECK:           %[[VAL_1:.*]] = emitc.call_opaque "acos"(%[[VAL_0]]) : (f32) -> f32
// CHECK:           return
// CHECK:         }
func.func @acos_to_call_opaque(%arg0: f32) {
    %1 = math.acos %arg0 : f32
    return
  }

// -----

// CHECK-LABEL:   func.func @atan2_to_call_opaque(
// CHECK-SAME:                                    %[[VAL_0:.*]]: f32,
// CHECK-SAME:                                    %[[VAL_1:.*]]: f32) {
// CHECK:           %[[VAL_2:.*]] = emitc.call_opaque "atan2"(%[[VAL_0]], %[[VAL_1]]) : (f32, f32) -> f32
// CHECK:           return
// CHECK:         }
func.func @atan2_to_call_opaque(%arg0: f32, %arg1: f32) {
    %1 = math.atan2 %arg0, %arg1 : f32
    return
  }

// -----

// CHECK-LABEL:   func.func @ceil_to_call_opaque(
// CHECK-SAME:                                   %[[VAL_0:.*]]: f32) {
// CHECK:           %[[VAL_1:.*]] = emitc.call_opaque "ceil"(%[[VAL_0]]) : (f32) -> f32
// CHECK:           return
// CHECK:         }
func.func @ceil_to_call_opaque(%arg0: f32) {
    %1 = math.ceil %arg0 : f32
    return
  }

// -----

// CHECK-LABEL:   func.func @exp_to_call_opaque(
// CHECK-SAME:                                  %[[VAL_0:.*]]: f32) {
// CHECK:           %[[VAL_1:.*]] = emitc.call_opaque "exp"(%[[VAL_0]]) : (f32) -> f32
// CHECK:           return
// CHECK:         }
func.func @exp_to_call_opaque(%arg0: f32) {
    %1 = math.exp %arg0 : f32
    return
  }


// -----

// CHECK-LABEL:   func.func @fpowi_to_call_opaque(
// CHECK-SAME:                                    %[[VAL_0:.*]]: f32,
// CHECK-SAME:                                    %[[VAL_1:.*]]: i32) {
// CHECK:           %[[VAL_2:.*]] = emitc.call_opaque "powf"(%[[VAL_0]], %[[VAL_1]]) : (f32, i32) -> f32
// CHECK:           return
// CHECK:         }
func.func @fpowi_to_call_opaque(%arg0: f32, %arg1: i32) {
    %1 = math.fpowi %arg0, %arg1 : f32, i32
    return
  }

// -----

// CHECK-LABEL:   func.func @ipowi_to_call_opaque(
// CHECK-SAME:                                    %[[VAL_0:.*]]: i32,
// CHECK-SAME:                                    %[[VAL_1:.*]]: i32) {
// CHECK:           %[[VAL_2:.*]] = emitc.call_opaque "pow"(%[[VAL_0]], %[[VAL_1]]) : (i32, i32) -> i32
// CHECK:           return
// CHECK:         }
func.func @ipowi_to_call_opaque(%arg0: i32, %arg1: i32) {
    %1 = math.ipowi %arg0, %arg1 : i32
    return
  }



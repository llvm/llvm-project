// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s -check-prefix=CPP-DEFAULT
// RUN: mlir-translate -mlir-to-cpp -declare-variables-at-top %s | FileCheck %s -check-prefix=CPP-DECLTOP

func.func @emitc_literal(%arg0: f32) {
  %p0 = emitc.literal "M_PI" : f32
  %1 = "emitc.add" (%arg0, %p0) : (f32, f32) -> f32
  return
}
// CPP-DEFAULT: void emitc_literal(float [[V0:[^ ]*]]) {
// CPP-DEFAULT: float [[V2:[^ ]*]] = [[V0:[^ ]*]] + M_PI

// CPP-DECLTOP: void emitc_literal(float [[V0:[^ ]*]]) {
// CPP-DECLTOP: float [[V1:[^ ]*]];
// CPP-DECLTOP: [[V1]] = [[V0:[^ ]*]] + M_PI

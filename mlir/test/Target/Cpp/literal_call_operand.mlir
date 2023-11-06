// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s -check-prefix=CPP-DEFAULT
// RUN: mlir-translate -mlir-to-cpp -declare-variables-at-top %s | FileCheck %s -check-prefix=CPP-DECLTOP

func.func @emitc_call_operand() {
  %p0 = emitc.literal "M_PI" : f32
  %1 = emitc.call "foo"(%p0) : (f32) -> f32
  return
}
// CPP-DEFAULT: void emitc_call_operand() {
// CPP-DEFAULT-NEXT: float v1 = foo(M_PI);

// CPP-DECLTOP: void emitc_call_operand() {
// CPP-DECLTOP-NEXT: float v1;
// CPP-DECLTOP-NEXT: v1 = foo(M_PI);

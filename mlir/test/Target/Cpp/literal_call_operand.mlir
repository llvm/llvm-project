// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s -check-prefix=CPP-DEFAULT
// RUN: mlir-translate -mlir-to-cpp -declare-variables-at-top %s | FileCheck %s -check-prefix=CPP-DECLTOP

func.func @emitc_call_operand() {
  %p0 = emitc.literal "M_PI" : f32
  %1 = emitc.call_opaque "foo"(%p0) : (f32) -> f32
  return
}
// CPP-DEFAULT: void emitc_call_operand() {
// CPP-DEFAULT-NEXT: float v1 = foo(M_PI);

// CPP-DECLTOP: void emitc_call_operand() {
// CPP-DECLTOP-NEXT: float v1;
// CPP-DECLTOP-NEXT: v1 = foo(M_PI);

func.func @emitc_call_operand_arg() {
  %p0 = emitc.literal "M_PI" : f32
  %1 = emitc.call_opaque "bar"(%p0) {args = [42 : i32, 0 : index]} : (f32) -> f32
  return
}
// CPP-DEFAULT: void emitc_call_operand_arg() {
// CPP-DEFAULT-NEXT: float v1 = bar(42, M_PI);

// CPP-DECLTOP: void emitc_call_operand_arg() {
// CPP-DECLTOP-NEXT: float v1;
// CPP-DECLTOP-NEXT: v1 = bar(42, M_PI);

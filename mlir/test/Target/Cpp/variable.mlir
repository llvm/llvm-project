// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s -check-prefix=CPP-DEFAULT
// RUN: mlir-translate -mlir-to-cpp -declare-variables-at-top %s | FileCheck %s -check-prefix=CPP-DECLTOP

func.func @emitc_variable() {
  %c0 = "emitc.variable"(){value = #emitc.opaque<""> : i32} : () -> i32
  %c1 = "emitc.variable"(){value = 42 : i32} : () -> i32
  %c2 = "emitc.variable"(){value = -1 : i32} : () -> i32
  %c3 = "emitc.variable"(){value = -1 : si8} : () -> si8
  %c4 = "emitc.variable"(){value = 255 : ui8} : () -> ui8
  %c5 = "emitc.variable"(){value = #emitc.opaque<""> : !emitc.ptr<i32>} : () -> !emitc.ptr<i32>
  %c6 = "emitc.variable"(){value = #emitc.opaque<"NULL"> : !emitc.ptr<i32>} : () -> !emitc.ptr<i32>
  return
}
// CPP-DEFAULT: void emitc_variable() {
// CPP-DEFAULT-NEXT: int32_t [[V0:[^ ]*]];
// CPP-DEFAULT-NEXT: int32_t [[V1:[^ ]*]] = 42;
// CPP-DEFAULT-NEXT: int32_t [[V2:[^ ]*]] = -1;
// CPP-DEFAULT-NEXT: int8_t [[V3:[^ ]*]] = -1;
// CPP-DEFAULT-NEXT: uint8_t [[V4:[^ ]*]] = 255;
// CPP-DEFAULT-NEXT: int32_t* [[V5:[^ ]*]];
// CPP-DEFAULT-NEXT: int32_t* [[V6:[^ ]*]] = NULL;

// CPP-DECLTOP: void emitc_variable() {
// CPP-DECLTOP-NEXT: int32_t [[V0:[^ ]*]];
// CPP-DECLTOP-NEXT: int32_t [[V1:[^ ]*]];
// CPP-DECLTOP-NEXT: int32_t [[V2:[^ ]*]];
// CPP-DECLTOP-NEXT: int8_t [[V3:[^ ]*]];
// CPP-DECLTOP-NEXT: uint8_t [[V4:[^ ]*]];
// CPP-DECLTOP-NEXT: int32_t* [[V5:[^ ]*]];
// CPP-DECLTOP-NEXT: int32_t* [[V6:[^ ]*]];
// CPP-DECLTOP-NEXT: ;
// CPP-DECLTOP-NEXT: [[V1]] = 42;
// CPP-DECLTOP-NEXT: [[V2]] = -1;
// CPP-DECLTOP-NEXT: [[V3]] = -1;
// CPP-DECLTOP-NEXT: [[V4]] = 255;
// CPP-DECLTOP-NEXT: ;
// CPP-DECLTOP-NEXT: [[V6]] = NULL;

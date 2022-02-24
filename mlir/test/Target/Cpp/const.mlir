// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s -check-prefix=CPP-DEFAULT
// RUN: mlir-translate -mlir-to-cpp -declare-variables-at-top %s | FileCheck %s -check-prefix=CPP-DECLTOP

func @emitc_constant() {
  %c0 = "emitc.constant"(){value = #emitc.opaque<""> : i32} : () -> i32
  %c1 = "emitc.constant"(){value = 42 : i32} : () -> i32
  %c2 = "emitc.constant"(){value = -1 : i32} : () -> i32
  %c3 = "emitc.constant"(){value = -1 : si8} : () -> si8
  %c4 = "emitc.constant"(){value = 255 : ui8} : () -> ui8
  %c5 = "emitc.constant"(){value = #emitc.opaque<"CHAR_MIN"> : !emitc.opaque<"char">} : () -> !emitc.opaque<"char">
  return
}
// CPP-DEFAULT: void emitc_constant() {
// CPP-DEFAULT-NEXT: int32_t [[V0:[^ ]*]];
// CPP-DEFAULT-NEXT: int32_t [[V1:[^ ]*]] = 42;
// CPP-DEFAULT-NEXT: int32_t [[V2:[^ ]*]] = -1;
// CPP-DEFAULT-NEXT: int8_t [[V3:[^ ]*]] = -1;
// CPP-DEFAULT-NEXT: uint8_t [[V4:[^ ]*]] = 255;
// CPP-DEFAULT-NEXT: char [[V5:[^ ]*]] = CHAR_MIN;

// CPP-DECLTOP: void emitc_constant() {
// CPP-DECLTOP-NEXT: int32_t [[V0:[^ ]*]];
// CPP-DECLTOP-NEXT: int32_t [[V1:[^ ]*]];
// CPP-DECLTOP-NEXT: int32_t [[V2:[^ ]*]];
// CPP-DECLTOP-NEXT: int8_t [[V3:[^ ]*]];
// CPP-DECLTOP-NEXT: uint8_t [[V4:[^ ]*]];
// CPP-DECLTOP-NEXT: char [[V5:[^ ]*]];
// CPP-DECLTOP-NEXT: ;
// CPP-DECLTOP-NEXT: [[V1]] = 42;
// CPP-DECLTOP-NEXT: [[V2]] = -1;
// CPP-DECLTOP-NEXT: [[V3]] = -1;
// CPP-DECLTOP-NEXT: [[V4]] = 255;
// CPP-DECLTOP-NEXT: [[V5]] = CHAR_MIN;

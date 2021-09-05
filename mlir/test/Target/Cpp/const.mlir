// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s -check-prefix=CPP-DEFAULT
// RUN: mlir-translate -mlir-to-cpp -declare-variables-at-top %s | FileCheck %s -check-prefix=CPP-DECLTOP


func @emitc_constant() {
  %c0 = "emitc.constant"(){value = #emitc.opaque<""> : i32} : () -> i32
  %c1 = "emitc.constant"(){value = 42 : i32} : () -> i32
  %c2 = "emitc.constant"(){value = #emitc.opaque<""> : !emitc.opaque<"int32_t*">} : () -> !emitc.opaque<"int32_t*">
  %c3 = "emitc.constant"(){value = #emitc.opaque<"NULL"> : !emitc.opaque<"int32_t*">} : () -> !emitc.opaque<"int32_t*">
  return
}
// CPP-DEFAULT: void emitc_constant() {
// CPP-DEFAULT-NEXT: int32_t [[V0:[^ ]*]];
// CPP-DEFAULT-NEXT: int32_t [[V1:[^ ]*]] = 42;
// CPP-DEFAULT-NEXT: int32_t* [[V2:[^ ]*]];
// CPP-DEFAULT-NEXT: int32_t* [[V3:[^ ]*]] = NULL;

// CPP-DECLTOP: void emitc_constant() {
// CPP-DECLTOP-NEXT: int32_t [[V0:[^ ]*]];
// CPP-DECLTOP-NEXT: int32_t [[V1:[^ ]*]];
// CPP-DECLTOP-NEXT: int32_t* [[V2:[^ ]*]];
// CPP-DECLTOP-NEXT: int32_t* [[V3:[^ ]*]];
// CPP-DECLTOP-NEXT: ;
// CPP-DECLTOP-NEXT: [[V1]] = 42;
// CPP-DECLTOP-NEXT: ;
// CPP-DECLTOP-NEXT: [[V3]] = NULL;

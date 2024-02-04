// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s -check-prefix=CPP-DEFAULT
// RUN: mlir-translate -mlir-to-cpp -declare-variables-at-top %s | FileCheck %s -check-prefix=CPP-DECLTOP


emitc.func @emitc_func(%arg0 : i32) {
  emitc.call_opaque "foo" (%arg0) : (i32) -> ()
  emitc.return
}
// CPP-DEFAULT: void emitc_func(int32_t [[V0:[^ ]*]]) {
// CPP-DEFAULT-NEXT: foo([[V0:[^ ]*]]);
// CPP-DEFAULT-NEXT: return;


emitc.func @return_i32() -> i32 attributes {specifiers = ["static","inline"]} {
  %0 = emitc.call_opaque "foo" (): () -> i32
  emitc.return %0 : i32
}
// CPP-DEFAULT: static inline int32_t return_i32() {
// CPP-DEFAULT-NEXT: [[V0:[^ ]*]] = foo();
// CPP-DEFAULT-NEXT: return [[V0:[^ ]*]];

// CPP-DECLTOP: static inline int32_t return_i32() {
// CPP-DECLTOP-NEXT: int32_t [[V0:[^ ]*]];
// CPP-DECLTOP-NEXT: [[V0:]] = foo();
// CPP-DECLTOP-NEXT: return [[V0:[^ ]*]];


emitc.func @emitc_call() -> i32 {
  %0 = emitc.call @return_i32() : () -> (i32)
  emitc.return %0 : i32
}
// CPP-DEFAULT: int32_t emitc_call() {
// CPP-DEFAULT-NEXT: int32_t [[V0:[^ ]*]] = return_i32();
// CPP-DEFAULT-NEXT: return [[V0:[^ ]*]];

// CPP-DECLTOP: int32_t emitc_call() {
// CPP-DECLTOP-NEXT: int32_t [[V0:[^ ]*]];
// CPP-DECLTOP-NEXT: [[V0:[^ ]*]] = return_i32();
// CPP-DECLTOP-NEXT: return [[V0:[^ ]*]];

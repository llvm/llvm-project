// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s -check-prefix=CPP-DEFAULT
// RUN: mlir-translate -mlir-to-cpp -declare-variables-at-top %s | FileCheck %s -check-prefix=CPP-DECLTOP

func.func @emitc_call_opaque() {
  %0 = emitc.call_opaque "func_a" () : () -> i32
  %1 = emitc.call_opaque "func_b" () : () -> i32
  return
}
// CPP-DEFAULT: void emitc_call_opaque() {
// CPP-DEFAULT-NEXT: int32_t [[V0:[^ ]*]] = func_a();
// CPP-DEFAULT-NEXT: int32_t [[V1:[^ ]*]] = func_b();

// CPP-DECLTOP: void emitc_call_opaque() {
// CPP-DECLTOP-NEXT: int32_t [[V0:[^ ]*]];
// CPP-DECLTOP-NEXT: int32_t [[V1:[^ ]*]];
// CPP-DECLTOP-NEXT: [[V0:]] = func_a();
// CPP-DECLTOP-NEXT: [[V1:]] = func_b();


func.func @emitc_call_opaque_two_results() {
  %0 = "emitc.constant"() <{value = 0 : index}> : () -> index
  %1:2 = emitc.call_opaque "two_results" () : () -> (i32, i32)
  return
}
// CPP-DEFAULT: void emitc_call_opaque_two_results() {
// CPP-DEFAULT-NEXT: size_t [[V1:[^ ]*]] = 0;
// CPP-DEFAULT-NEXT: int32_t [[V2:[^ ]*]];
// CPP-DEFAULT-NEXT: int32_t [[V3:[^ ]*]];
// CPP-DEFAULT-NEXT: std::tie([[V2]], [[V3]]) = two_results();

// CPP-DECLTOP: void emitc_call_opaque_two_results() {
// CPP-DECLTOP-NEXT: size_t [[V1:[^ ]*]];
// CPP-DECLTOP-NEXT: int32_t [[V2:[^ ]*]];
// CPP-DECLTOP-NEXT: int32_t [[V3:[^ ]*]];
// CPP-DECLTOP-NEXT: [[V1]] = 0;
// CPP-DECLTOP-NEXT: std::tie([[V2]], [[V3]]) = two_results();

func.func @emitc_call_opaque_member(%arg0 : !emitc.opaque<"MyClass">, %arg1 : !emitc.ptr<!emitc.opaque<"MyClass">>) {
  %0 = emitc.call_opaque "method" (%arg0) {is_member_call = true} : (!emitc.opaque<"MyClass">) -> i32
  %1 = emitc.call_opaque "ptr_method" (%arg1) {is_member_call = true} : (!emitc.ptr<!emitc.opaque<"MyClass">>) -> i32
  return
}
// CPP-DEFAULT: void emitc_call_opaque_member(MyClass [[V0:[^ ]*]], MyClass* [[V1:[^ ]*]]) {
// CPP-DEFAULT-NEXT: int32_t [[V2:[^ ]*]] = [[V0]].method();
// CPP-DEFAULT-NEXT: int32_t [[V3:[^ ]*]] = [[V1]]->ptr_method();

func.func @emitc_call_opaque_member_args(%arg0 : !emitc.opaque<"MyClass">, %arg1 : i32, %arg2 : i32) {
  %0 = emitc.call_opaque "method" (%arg0, %arg1, %arg2) {is_member_call = true} : (!emitc.opaque<"MyClass">, i32, i32) -> i32
  %1 = emitc.call_opaque "method_with_args" (%arg0, %arg1, %arg2) {is_member_call = true, args = [1 : index, 2 : index]} : (!emitc.opaque<"MyClass">, i32, i32) -> i32
  return
}
// CPP-DEFAULT: void emitc_call_opaque_member_args(MyClass [[V0:[^ ]*]], int32_t [[V1:[^ ]*]], int32_t [[V2:[^ ]*]]) {
// CPP-DEFAULT-NEXT: int32_t [[V3:[^ ]*]] = [[V0]].method([[V1]], [[V2]]);
// CPP-DEFAULT-NEXT: int32_t [[V4:[^ ]*]] = [[V0]].method_with_args([[V1]], [[V2]]);

// CPP-DECLTOP: void emitc_call_opaque_member_args(MyClass [[V0:[^ ]*]], int32_t [[V1:[^ ]*]], int32_t [[V2:[^ ]*]]) {
// CPP-DECLTOP-NEXT: int32_t [[V3:[^ ]*]];
// CPP-DECLTOP-NEXT: int32_t [[V4:[^ ]*]];
// CPP-DECLTOP-NEXT: [[V3]] = [[V0]].method([[V1]], [[V2]]);
// CPP-DECLTOP-NEXT: [[V4]] = [[V0]].method_with_args([[V1]], [[V2]]);

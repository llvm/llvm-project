// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s -check-prefix=CPP-DEFAULT

func.func @member(%arg0: !emitc.opaque<"mystruct">, %arg1: i32) {
  %0 = "emitc.member" (%arg0) {member = "a"} : (!emitc.opaque<"mystruct">) -> i32
  emitc.assign %arg1 : i32 to %0 : i32 

  %1 = "emitc.member" (%arg0) {member = "b"} : (!emitc.opaque<"mystruct">) -> i32
  %2 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> i32
  emitc.assign %1 : i32 to %2 : i32

  return
}

// CPP-DEFAULT: void member(mystruct [[V0:[^ ]*]], int32_t [[V1:[^ ]*]]) {
// CPP-DEFAULT-NEXT: [[V0:[^ ]*]].a = [[V1:[^ ]*]];
// CPP-DEFAULT-NEXT: int32_t [[V2:[^ ]*]];
// CPP-DEFAULT-NEXT: [[V2:[^ ]*]] = [[V0:[^ ]*]].b;


func.func @member_of_pointer(%arg0: !emitc.ptr<!emitc.opaque<"mystruct">>, %arg1: i32) {
  %0 = "emitc.member_of_ptr" (%arg0) {member = "a"} : (!emitc.ptr<!emitc.opaque<"mystruct">>) -> i32
  emitc.assign %arg1 : i32 to %0 : i32

  %1 = "emitc.member_of_ptr" (%arg0) {member = "b"} : (!emitc.ptr<!emitc.opaque<"mystruct">>) -> i32
  %2 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> i32
  emitc.assign %1 : i32 to %2 : i32

  return
}

// CPP-DEFAULT: void member_of_pointer(mystruct* [[V0:[^ ]*]], int32_t [[V1:[^ ]*]]) {
// CPP-DEFAULT-NEXT: [[V0:[^ ]*]]->a = [[V1:[^ ]*]];
// CPP-DEFAULT-NEXT: int32_t [[V2:[^ ]*]];
// CPP-DEFAULT-NEXT: [[V2:[^ ]*]] = [[V0:[^ ]*]]->b;

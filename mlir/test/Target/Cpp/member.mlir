// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s -check-prefix=CPP-DEFAULT

func.func @member(%arg0: !emitc.lvalue<!emitc.opaque<"mystruct">>, %arg1: i32) {
  %0 = "emitc.member" (%arg0) {member = "a"} : (!emitc.lvalue<!emitc.opaque<"mystruct">>) -> !emitc.lvalue<i32>
  emitc.assign %arg1 : i32 to %0 : !emitc.lvalue<i32> 

  %1 = "emitc.member" (%arg0) {member = "b"} : (!emitc.lvalue<!emitc.opaque<"mystruct">>) -> !emitc.lvalue<i32>
  %2 = emitc.lvalue_load %1 : !emitc.lvalue<i32>
  %3 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
  emitc.assign %2 : i32 to %3 : !emitc.lvalue<i32>

  return
}

// CPP-DEFAULT: void member(mystruct [[V0:[^ ]*]], int32_t [[V1:[^ ]*]]) {
// CPP-DEFAULT-NEXT: [[V0]].a = [[V1]];
// CPP-DEFAULT-NEXT: int32_t [[V2:[^ ]*]] = [[V0]].b;
// CPP-DEFAULT-NEXT: int32_t [[V3:[^ ]*]];
// CPP-DEFAULT-NEXT: [[V3]] = [[V2]];


func.func @member_of_pointer(%arg0: !emitc.lvalue<!emitc.ptr<!emitc.opaque<"mystruct">>>, %arg1: i32) {
  %0 = "emitc.member_of_ptr" (%arg0) {member = "a"} : (!emitc.lvalue<!emitc.ptr<!emitc.opaque<"mystruct">>>) -> !emitc.lvalue<i32>
  emitc.assign %arg1 : i32 to %0 : !emitc.lvalue<i32>

  %1 = "emitc.member_of_ptr" (%arg0) {member = "b"} : (!emitc.lvalue<!emitc.ptr<!emitc.opaque<"mystruct">>>) -> !emitc.lvalue<i32>
  %2 = emitc.lvalue_load %1 : !emitc.lvalue<i32>
  %3 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
  emitc.assign %2 : i32 to %3 : !emitc.lvalue<i32>

  return
}

// CPP-DEFAULT: void member_of_pointer(mystruct* [[V0:[^ ]*]], int32_t [[V1:[^ ]*]]) {
// CPP-DEFAULT-NEXT: [[V0]]->a = [[V1]];
// CPP-DEFAULT-NEXT: int32_t [[V2:[^ ]*]] = [[V0]]->b;
// CPP-DEFAULT-NEXT: int32_t [[V3:[^ ]*]];
// CPP-DEFAULT-NEXT: [[V3]] = [[V2]];


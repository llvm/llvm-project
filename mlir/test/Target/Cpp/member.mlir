// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s -check-prefix=CPP-DEFAULT

func.func @member(%arg0: !emitc.opaque<"mystruct">, %arg1: i32) {
  %var0 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"mystruct">>
  emitc.assign %arg0 : !emitc.opaque<"mystruct"> to %var0 : !emitc.lvalue<!emitc.opaque<"mystruct">>

  %0 = "emitc.member" (%var0) {member = "a"} : (!emitc.lvalue<!emitc.opaque<"mystruct">>) -> !emitc.lvalue<i32>
  emitc.assign %arg1 : i32 to %0 : !emitc.lvalue<i32> 

  %1 = "emitc.member" (%var0) {member = "b"} : (!emitc.lvalue<!emitc.opaque<"mystruct">>) -> !emitc.lvalue<i32>
  %2 = emitc.load %1 : !emitc.lvalue<i32>
  %3 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
  emitc.assign %2 : i32 to %3 : !emitc.lvalue<i32>

  return
}

// CPP-DEFAULT: void member(mystruct [[V0:[^ ]*]], int32_t [[V1:[^ ]*]]) {
// CPP-DEFAULT-NEXT: mystruct [[V2:[^ ]*]];
// CPP-DEFAULT-NEXT: [[V2]] = [[V0]];
// CPP-DEFAULT-NEXT: [[V2]].a = [[V1]];
// CPP-DEFAULT-NEXT: int32_t [[V3:[^ ]*]] = [[V2]].b;
// CPP-DEFAULT-NEXT: int32_t [[V4:[^ ]*]];
// CPP-DEFAULT-NEXT: [[V4]] = [[V3]];


func.func @member_of_pointer(%arg0: !emitc.ptr<!emitc.opaque<"mystruct">>, %arg1: i32) {
  %var0 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.ptr<!emitc.opaque<"mystruct">>>
  emitc.assign %arg0 : !emitc.ptr<!emitc.opaque<"mystruct">> to %var0 : !emitc.lvalue<!emitc.ptr<!emitc.opaque<"mystruct">>>
  
  %0 = "emitc.member_of_ptr" (%var0) {member = "a"} : (!emitc.lvalue<!emitc.ptr<!emitc.opaque<"mystruct">>>) -> !emitc.lvalue<i32>
  emitc.assign %arg1 : i32 to %0 : !emitc.lvalue<i32>

  %1 = "emitc.member_of_ptr" (%var0) {member = "b"} : (!emitc.lvalue<!emitc.ptr<!emitc.opaque<"mystruct">>>) -> !emitc.lvalue<i32>
  %2 = emitc.load %1 : !emitc.lvalue<i32>
  %3 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
  emitc.assign %2 : i32 to %3 : !emitc.lvalue<i32>

  return
}

// CPP-DEFAULT: void member_of_pointer(mystruct* [[V0:[^ ]*]], int32_t [[V1:[^ ]*]]) {
// CPP-DEFAULT-NEXT: mystruct* [[V2:[^ ]*]];
// CPP-DEFAULT-NEXT: [[V2]] = [[V0]];
// CPP-DEFAULT-NEXT: [[V2]]->a = [[V1]];
// CPP-DEFAULT-NEXT: int32_t [[V3:[^ ]*]] = [[V2]]->b;
// CPP-DEFAULT-NEXT: int32_t [[V4:[^ ]*]];
// CPP-DEFAULT-NEXT: [[V4]] = [[V3]];


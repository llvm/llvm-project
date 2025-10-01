// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s -check-prefix=CPP-DEFAULT

func.func @member(%arg0: !emitc.opaque<"mystruct">, %arg1: i32, %arg2: index) {
  %var0 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"mystruct">>
  emitc.assign %arg0 : !emitc.opaque<"mystruct"> to %var0 : !emitc.lvalue<!emitc.opaque<"mystruct">>

  %0 = "emitc.member" (%var0) {member = "a"} : (!emitc.lvalue<!emitc.opaque<"mystruct">>) -> !emitc.lvalue<i32>
  emitc.assign %arg1 : i32 to %0 : !emitc.lvalue<i32> 

  %1 = "emitc.member" (%var0) {member = "b"} : (!emitc.lvalue<!emitc.opaque<"mystruct">>) -> !emitc.lvalue<i32>
  %2 = emitc.load %1 : !emitc.lvalue<i32>
  %3 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
  emitc.assign %2 : i32 to %3 : !emitc.lvalue<i32>

  %4 = "emitc.member" (%var0) {member = "c"} : (!emitc.lvalue<!emitc.opaque<"mystruct">>) -> !emitc.array<2xi32>
  %5 = emitc.subscript %4[%arg2] : (!emitc.array<2xi32>, index) -> !emitc.lvalue<i32>
  %6 = emitc.load %5 : <i32>
  emitc.assign %6 : i32 to %3 : !emitc.lvalue<i32>

  %7 = "emitc.member" (%var0) {member = "d"} : (!emitc.lvalue<!emitc.opaque<"mystruct">>) -> !emitc.array<2xi32>
  %8 = emitc.subscript %7[%arg2] : (!emitc.array<2xi32>, index) -> !emitc.lvalue<i32>
  emitc.assign %arg1 : i32 to %8 : !emitc.lvalue<i32>

  return
}

// CPP-DEFAULT: void member(mystruct [[V0:[^ ]*]], int32_t [[V1:[^ ]*]], size_t [[Index:[^ ]*]]) {
// CPP-DEFAULT-NEXT: mystruct [[V2:[^ ]*]];
// CPP-DEFAULT-NEXT: [[V2]] = [[V0]];
// CPP-DEFAULT-NEXT: [[V2]].a = [[V1]];
// CPP-DEFAULT-NEXT: int32_t [[V3:[^ ]*]] = [[V2]].b;
// CPP-DEFAULT-NEXT: int32_t [[V4:[^ ]*]];
// CPP-DEFAULT-NEXT: [[V4]] = [[V3]];
// CPP-DEFAULT-NEXT: int32_t [[V5:[^ ]*]] = [[V2]].c[[[Index]]];
// CPP-DEFAULT-NEXT: [[V4]] = [[V5]];
// CPP-DEFAULT-NEXT: [[V2]].d[[[Index]]] = [[V1]];


func.func @member_of_pointer(%arg0: !emitc.ptr<!emitc.opaque<"mystruct">>, %arg1: i32, %arg2: index) {
  %var0 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.ptr<!emitc.opaque<"mystruct">>>
  emitc.assign %arg0 : !emitc.ptr<!emitc.opaque<"mystruct">> to %var0 : !emitc.lvalue<!emitc.ptr<!emitc.opaque<"mystruct">>>
  
  %0 = "emitc.member_of_ptr" (%var0) {member = "a"} : (!emitc.lvalue<!emitc.ptr<!emitc.opaque<"mystruct">>>) -> !emitc.lvalue<i32>
  emitc.assign %arg1 : i32 to %0 : !emitc.lvalue<i32>

  %1 = "emitc.member_of_ptr" (%var0) {member = "b"} : (!emitc.lvalue<!emitc.ptr<!emitc.opaque<"mystruct">>>) -> !emitc.lvalue<i32>
  %2 = emitc.load %1 : !emitc.lvalue<i32>
  %3 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
  emitc.assign %2 : i32 to %3 : !emitc.lvalue<i32>

  %4 = "emitc.member_of_ptr" (%var0) {member = "c"} : (!emitc.lvalue<!emitc.ptr<!emitc.opaque<"mystruct">>>) -> !emitc.array<2xi32>
  %5 = emitc.subscript %4[%arg2] : (!emitc.array<2xi32>, index) -> !emitc.lvalue<i32>
  %6 = emitc.load %5 : <i32>
  emitc.assign %6 : i32 to %3 : !emitc.lvalue<i32>

  %7 = "emitc.member_of_ptr" (%var0) {member = "d"} : (!emitc.lvalue<!emitc.ptr<!emitc.opaque<"mystruct">>>) -> !emitc.array<2xi32>
  %8 = emitc.subscript %7[%arg2] : (!emitc.array<2xi32>, index) -> !emitc.lvalue<i32>
  emitc.assign %arg1 : i32 to %8 : !emitc.lvalue<i32>

  return
}

// CPP-DEFAULT: void member_of_pointer(mystruct* [[V0:[^ ]*]], int32_t [[V1:[^ ]*]], size_t [[Index:[^ ]*]]) {
// CPP-DEFAULT-NEXT: mystruct* [[V2:[^ ]*]];
// CPP-DEFAULT-NEXT: [[V2]] = [[V0]];
// CPP-DEFAULT-NEXT: [[V2]]->a = [[V1]];
// CPP-DEFAULT-NEXT: int32_t [[V3:[^ ]*]] = [[V2]]->b;
// CPP-DEFAULT-NEXT: int32_t [[V4:[^ ]*]];
// CPP-DEFAULT-NEXT: [[V4]] = [[V3]];
// CPP-DEFAULT-NEXT: int32_t [[V5:[^ ]*]] = [[V2]]->c[[[Index]]];
// CPP-DEFAULT-NEXT: [[V4]] = [[V5]];
// CPP-DEFAULT-NEXT: [[V2]]->d[[[Index]]] = [[V1]];

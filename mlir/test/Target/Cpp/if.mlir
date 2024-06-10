// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s -check-prefix=CPP-DEFAULT
// RUN: mlir-translate -mlir-to-cpp -declare-variables-at-top %s | FileCheck %s -check-prefix=CPP-DECLTOP

func.func @test_if(%arg0: i1, %arg1: f32) {
  emitc.if %arg0 {
     %0 = emitc.call_opaque "func_const"(%arg1) : (f32) -> i32
  }
  return
}
// CPP-DEFAULT: void test_if(bool [[V0:[^ ]*]], float [[V1:[^ ]*]]) {
// CPP-DEFAULT-NEXT: if ([[V0]]) {
// CPP-DEFAULT-NEXT: int32_t [[V2:[^ ]*]] = func_const([[V1]]);
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-NEXT: return;

// CPP-DECLTOP: void test_if(bool [[V0:[^ ]*]], float [[V1:[^ ]*]]) {
// CPP-DECLTOP-NEXT: int32_t [[V2:[^ ]*]];
// CPP-DECLTOP-NEXT: if ([[V0]]) {
// CPP-DECLTOP-NEXT: [[V2]] = func_const([[V1]]);
// CPP-DECLTOP-NEXT: }
// CPP-DECLTOP-NEXT: return;


func.func @test_if_else(%arg0: i1, %arg1: f32) {
  emitc.if %arg0 {
    %0 = emitc.call_opaque "func_true"(%arg1) : (f32) -> i32
  } else {
    %0 = emitc.call_opaque "func_false"(%arg1) : (f32) -> i32
  }
  return
}
// CPP-DEFAULT: void test_if_else(bool [[V0:[^ ]*]], float [[V1:[^ ]*]]) {
// CPP-DEFAULT-NEXT: if ([[V0]]) {
// CPP-DEFAULT-NEXT: int32_t [[V2:[^ ]*]] = func_true([[V1]]);
// CPP-DEFAULT-NEXT: } else {
// CPP-DEFAULT-NEXT: int32_t [[V3:[^ ]*]] = func_false([[V1]]);
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-NEXT: return;

// CPP-DECLTOP: void test_if_else(bool [[V0:[^ ]*]], float [[V1:[^ ]*]]) {
// CPP-DECLTOP-NEXT: int32_t [[V2:[^ ]*]];
// CPP-DECLTOP-NEXT: int32_t [[V3:[^ ]*]];
// CPP-DECLTOP-NEXT: if ([[V0]]) {
// CPP-DECLTOP-NEXT: [[V2]] = func_true([[V1]]);
// CPP-DECLTOP-NEXT: } else {
// CPP-DECLTOP-NEXT: [[V3]] = func_false([[V1]]);
// CPP-DECLTOP-NEXT: }
// CPP-DECLTOP-NEXT: return;


func.func @test_if_yield(%arg0: i1, %arg1: f32) {
  %0 = "emitc.constant"() <{value = 0 : i8}> : () -> i8
  %1 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<1xi32>
  %2 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<1xf64>
  emitc.if %arg0 {
    %8 = emitc.call_opaque "func_true_1"(%arg1) : (f32) -> i32
    %9 = emitc.call_opaque "func_true_2"(%arg1) : (f32) -> f64
    %10 = "emitc.constant"() <{value = 0 : index}> : () -> index
    %11 = emitc.subscript %1[%10] : (!emitc.array<1xi32>, index) -> i32
    emitc.assign %8 : i32 to %11 : i32
    %12 = emitc.subscript %2[%10] : (!emitc.array<1xf64>, index) -> f64
    emitc.assign %9 : f64 to %12 : f64
  } else {
    %8 = emitc.call_opaque "func_false_1"(%arg1) : (f32) -> i32
    %9 = emitc.call_opaque "func_false_2"(%arg1) : (f32) -> f64
    %10 = "emitc.constant"() <{value = 0 : index}> : () -> index
    %11 = emitc.subscript %1[%10] : (!emitc.array<1xi32>, index) -> i32
    emitc.assign %8 : i32 to %11 : i32
    %12 = emitc.subscript %2[%10] : (!emitc.array<1xf64>, index) -> f64
    emitc.assign %9 : f64 to %12 : f64
  }
  %3 = "emitc.constant"() <{value = 0 : index}> : () -> index
  %4 = emitc.subscript %1[%3] : (!emitc.array<1xi32>, index) -> i32
  %5 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> i32
  emitc.assign %4 : i32 to %5 : i32
  %6 = emitc.subscript %2[%3] : (!emitc.array<1xf64>, index) -> f64
  %7 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f64
  emitc.assign %6 : f64 to %7 : f64
  return
}
// CPP-DEFAULT: void test_if_yield(bool [[V1:[^ ]*]], float [[V2:[^ ]*]]) {
// CPP-DEFAULT-NEXT: int8_t [[V3:[^ ]*]] = 0;
// CPP-DEFAULT-NEXT: int32_t [[V4:[^ ]*]][1];
// CPP-DEFAULT-NEXT: double [[V5:[^ ]*]][1];
// CPP-DEFAULT-NEXT: if ([[V1]]) {
// CPP-DEFAULT-NEXT:   int32_t [[V6:[^ ]*]] = func_true_1([[V2]]);
// CPP-DEFAULT-NEXT:   double [[V7:[^ ]*]] = func_true_2([[V2]]);
// CPP-DEFAULT-NEXT:   size_t [[V8:[^ ]*]] = 0;
// CPP-DEFAULT-NEXT:   [[V4]][[[V8]]] = [[V6]];
// CPP-DEFAULT-NEXT:   [[V5]][[[V8]]] = [[V7]];
// CPP-DEFAULT-NEXT: } else {
// CPP-DEFAULT-NEXT:   int32_t [[V9:[^ ]*]] = func_false_1([[V2]]);
// CPP-DEFAULT-NEXT:   double [[V10:[^ ]*]] = func_false_2([[V2]]);
// CPP-DEFAULT-NEXT:   size_t [[V11:[^ ]*]] = 0;
// CPP-DEFAULT-NEXT:   [[V4]][[[V11]]] = [[V9]];
// CPP-DEFAULT-NEXT:   [[V5]][[[V11]]] = [[V10]];
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-NEXT: size_t [[V12:[^ ]*]] = 0;
// CPP-DEFAULT-NEXT: int32_t [[V13:[^ ]*]];
// CPP-DEFAULT-NEXT: [[V13]] = [[V4]][[[V12]]];
// CPP-DEFAULT-NEXT: double [[V14:[^ ]*]];
// CPP-DEFAULT-NEXT: [[V14]] = [[V5]][[[V12]]];
// CPP-DEFAULT-NEXT: return;

// CPP-DECLTOP: void test_if_yield(bool [[V1:[^ ]*]], float [[V2:[^ ]*]]) {
// CPP-DECLTOP-NEXT: int8_t [[V3:[^ ]*]];
// CPP-DECLTOP-NEXT: int32_t [[V4:[^ ]*]][1];
// CPP-DECLTOP-NEXT: double [[V5:[^ ]*]][1];
// CPP-DECLTOP-NEXT: int32_t [[V6:[^ ]*]];
// CPP-DECLTOP-NEXT: double [[V7:[^ ]*]];
// CPP-DECLTOP-NEXT: size_t [[V8:[^ ]*]];
// CPP-DECLTOP-NEXT: int32_t [[V9:[^ ]*]];
// CPP-DECLTOP-NEXT: double [[V10:[^ ]*]];
// CPP-DECLTOP-NEXT: size_t [[V11:[^ ]*]];
// CPP-DECLTOP-NEXT: size_t [[V12:[^ ]*]];
// CPP-DECLTOP-NEXT: int32_t [[V13:[^ ]*]];
// CPP-DECLTOP-NEXT: double [[V14:[^ ]*]];
// CPP-DECLTOP-NEXT: [[V3]] = 0;
// CPP-DECLTOP-NEXT: ;
// CPP-DECLTOP-NEXT: ;
// CPP-DECLTOP-NEXT: if ([[V1]]) {
// CPP-DECLTOP-NEXT:   [[V6]] = func_true_1([[V2]]);
// CPP-DECLTOP-NEXT:   [[V7]] = func_true_2([[V2]]);
// CPP-DECLTOP-NEXT:   [[V8]] = 0;
// CPP-DECLTOP-NEXT:   [[V4]][[[V8]]] = [[V6]];
// CPP-DECLTOP-NEXT:   [[V5]][[[V8]]] = [[V7]];
// CPP-DECLTOP-NEXT: } else {
// CPP-DECLTOP-NEXT:   [[V9]] = func_false_1([[V2]]);
// CPP-DECLTOP-NEXT:   [[V10]] = func_false_2([[V2]]);
// CPP-DECLTOP-NEXT:   [[V11]] = 0;
// CPP-DECLTOP-NEXT:   [[V4]][[[V11]]] = [[V9]];
// CPP-DECLTOP-NEXT:   [[V5]][[[V11]]] = [[V10]];
// CPP-DECLTOP-NEXT: }
// CPP-DECLTOP-NEXT: [[V12]] = 0;
// CPP-DECLTOP-NEXT: ;
// CPP-DECLTOP-NEXT: [[V13]] = [[V4]][[[V12]]];
// CPP-DECLTOP-NEXT: ;
// CPP-DECLTOP-NEXT: [[V14]] = [[V5]][[[V12]]];
// CPP-DECLTOP-NEXT: return;

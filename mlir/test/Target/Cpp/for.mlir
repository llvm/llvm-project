// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s -check-prefix=CPP-DEFAULT
// RUN: mlir-translate -mlir-to-cpp -declare-variables-at-top %s | FileCheck %s -check-prefix=CPP-DECLTOP

func.func @test_for(%arg0 : index, %arg1 : index, %arg2 : index) {
  %lb = emitc.expression : index {
    %a = emitc.add %arg0, %arg1 : (index, index) -> index
    emitc.yield %a : index
  }
  %ub = emitc.expression : index {
    %a = emitc.mul %arg1, %arg2 : (index, index) -> index
    emitc.yield %a : index
  }
  %step = emitc.expression : index {
    %a = emitc.div %arg0, %arg2 : (index, index) -> index
    emitc.yield %a : index
  }
  emitc.for %i0 = %lb to %ub step %step {
    %0 = emitc.call_opaque "f"() : () -> i32
  }
  return
}
// CPP-DEFAULT: void test_for(size_t [[V1:[^ ]*]], size_t [[V2:[^ ]*]], size_t [[V3:[^ ]*]]) {
// CPP-DEFAULT-NEXT: for (size_t [[ITER:[^ ]*]] = [[V1]] + [[V2]]; [[ITER]] < ([[V2]] * [[V3]]); [[ITER]] += [[V1]] / [[V3]]) {
// CPP-DEFAULT-NEXT: int32_t [[V4:[^ ]*]] = f();
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-NEXT: return;

// CPP-DECLTOP: void test_for(size_t [[V1:[^ ]*]], size_t [[V2:[^ ]*]], size_t [[V3:[^ ]*]]) {
// CPP-DECLTOP-NEXT: int32_t [[V4:[^ ]*]];
// CPP-DECLTOP-NEXT: for (size_t [[ITER:[^ ]*]] = [[V1]] + [[V2]]; [[ITER]] < ([[V2]] * [[V3]]); [[ITER]] += [[V1]] / [[V3]]) {
// CPP-DECLTOP-NEXT: [[V4]] = f();
// CPP-DECLTOP-NEXT: }
// CPP-DECLTOP-NEXT: return;

func.func @test_for_yield(%arg0: index, %arg1: index, %arg2: index) {
  %0 = "emitc.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
  %1 = "emitc.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
  %2 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<1xf32>
  %3 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<1xf32>
  %4 = "emitc.constant"() <{value = 0 : index}> : () -> index
  %5 = emitc.subscript %2[%4] : (!emitc.array<1xf32>, index) -> f32
  emitc.assign %0 : f32 to %5 : f32
  %6 = emitc.subscript %3[%4] : (!emitc.array<1xf32>, index) -> f32
  emitc.assign %1 : f32 to %6 : f32
  emitc.for %arg3 = %arg0 to %arg1 step %arg2 {
    %12 = "emitc.constant"() <{value = 0 : index}> : () -> index
    %13 = emitc.subscript %2[%12] : (!emitc.array<1xf32>, index) -> f32
    %14 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
    emitc.assign %13 : f32 to %14 : f32
    %15 = emitc.subscript %3[%12] : (!emitc.array<1xf32>, index) -> f32
    %16 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
    emitc.assign %15 : f32 to %16 : f32
    %17 = emitc.add %14, %16 : (f32, f32) -> f32
    %18 = "emitc.constant"() <{value = 0 : index}> : () -> index
    %19 = emitc.subscript %2[%18] : (!emitc.array<1xf32>, index) -> f32
    emitc.assign %17 : f32 to %19 : f32
    %20 = emitc.subscript %3[%18] : (!emitc.array<1xf32>, index) -> f32
    emitc.assign %17 : f32 to %20 : f32
  }
  %7 = "emitc.constant"() <{value = 0 : index}> : () -> index
  %8 = emitc.subscript %2[%7] : (!emitc.array<1xf32>, index) -> f32
  %9 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
  emitc.assign %8 : f32 to %9 : f32
  %10 = emitc.subscript %3[%7] : (!emitc.array<1xf32>, index) -> f32
  %11 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
  emitc.assign %10 : f32 to %11 : f32
  return
}
// CPP-DEFAULT: void test_for_yield(size_t v1, size_t v2, size_t v3) {
// CPP-DEFAULT-NEXT: float [[V4:[^ ]*]] = 0.0e+00f;
// CPP-DEFAULT-NEXT: float [[V5:[^ ]*]] = 1.000000000e+00f;
// CPP-DEFAULT-NEXT: float [[V6:[^ ]*]][1];
// CPP-DEFAULT-NEXT: float [[V7:[^ ]*]][1];
// CPP-DEFAULT-NEXT: size_t [[V8:[^ ]*]] = 0;
// CPP-DEFAULT-NEXT: [[V6]][[[V8]]] = [[V4]];
// CPP-DEFAULT-NEXT: [[V7]][[[V8]]] = [[V5]];
// CPP-DEFAULT-NEXT: for (size_t [[V9:[^ ]*]] = [[V1]]; [[V9]] < [[V2]]; [[V9]] += [[V3]]) {
// CPP-DEFAULT-NEXT:   size_t [[V10:[^ ]*]] = 0;
// CPP-DEFAULT-NEXT:   float [[V11:[^ ]*]];
// CPP-DEFAULT-NEXT:   [[V11]] = [[V6]][[[V10]]];
// CPP-DEFAULT-NEXT:   float [[V12:[^ ]*]];
// CPP-DEFAULT-NEXT:   [[V12]] = [[V7]][[[V10]]];
// CPP-DEFAULT-NEXT:   float [[V13:[^ ]*]] = [[V11]] + [[V12]];
// CPP-DEFAULT-NEXT:   size_t [[V14:[^ ]*]] = 0;
// CPP-DEFAULT-NEXT:   [[V6]][[[V14]]] = [[V13]];
// CPP-DEFAULT-NEXT:   [[V7]][[[V14]]] = [[V13]];
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-NEXT: size_t [[V15:[^ ]*]] = 0;
// CPP-DEFAULT-NEXT: float [[V16:[^ ]*]];
// CPP-DEFAULT-NEXT: v16 = v6[v15];
// CPP-DEFAULT-NEXT: float [[V17:[^ ]*]];
// CPP-DEFAULT-NEXT: v17 = v7[v15];
// CPP-DEFAULT-NEXT: return;

// CPP-DECLTOP: void test_for_yield(size_t v1, size_t v2, size_t v3) {
// CPP-DECLTOP-NEXT: float [[V4:[^ ]*]];
// CPP-DECLTOP-NEXT: float [[V5:[^ ]*]];
// CPP-DECLTOP-NEXT: float [[V6:[^ ]*]][1];
// CPP-DECLTOP-NEXT: float [[V7:[^ ]*]][1];
// CPP-DECLTOP-NEXT: size_t [[V8:[^ ]*]];
// CPP-DECLTOP-NEXT: size_t [[V9:[^ ]*]];
// CPP-DECLTOP-NEXT: float [[V10:[^ ]*]];
// CPP-DECLTOP-NEXT: float [[V11:[^ ]*]];
// CPP-DECLTOP-NEXT: float [[V12:[^ ]*]];
// CPP-DECLTOP-NEXT: size_t [[V13:[^ ]*]];
// CPP-DECLTOP-NEXT: size_t [[V14:[^ ]*]];
// CPP-DECLTOP-NEXT: float [[V15:[^ ]*]];
// CPP-DECLTOP-NEXT: float [[V16:[^ ]*]];
// CPP-DECLTOP-NEXT: [[V4]] = 0.0e+00f;
// CPP-DECLTOP-NEXT: [[V5]] = 1.000000000e+00f;
// CPP-DECLTOP-NEXT: ;
// CPP-DECLTOP-NEXT: ;
// CPP-DECLTOP-NEXT: [[V8]] = 0;
// CPP-DECLTOP-NEXT: [[V6]][[[V8]]] = [[V4]];
// CPP-DECLTOP-NEXT: [[V7]][[[V8]]] = [[V5]];
// CPP-DECLTOP-NEXT: for (size_t [[V17:[^ ]*]] = [[V1]]; [[V17]] < [[V2]]; [[V17]] += [[V3]]) {
// CPP-DECLTOP-NEXT:   [[V9]] = 0;
// CPP-DECLTOP-NEXT:   ;
// CPP-DECLTOP-NEXT:   [[V10]] = [[V6]][[[V9]]];
// CPP-DECLTOP-NEXT:   ;
// CPP-DECLTOP-NEXT:   [[V11]] = [[V7]][[[V9]]];
// CPP-DECLTOP-NEXT:   [[V12]] = [[V10]] + [[V11]];
// CPP-DECLTOP-NEXT:   [[V13]] = 0;
// CPP-DECLTOP-NEXT:   [[V6]][[[V13]]] = [[V12]];
// CPP-DECLTOP-NEXT:   [[V7]][[[V13]]] = [[V12]];
// CPP-DECLTOP-NEXT: }
// CPP-DECLTOP-NEXT: [[V14]] = 0;
// CPP-DECLTOP-NEXT: ;
// CPP-DECLTOP-NEXT: [[V15]] = [[V6]][[[V14]]];
// CPP-DECLTOP-NEXT: ;
// CPP-DECLTOP-NEXT: [[V16]] = [[V7]][[[V14]]];
// CPP-DECLTOP-NEXT: return;

func.func @test_for_yield_2() {
  %start = emitc.literal "0" : index
  %stop = emitc.literal "10" : index
  %step = emitc.literal "1" : index

  %0 = emitc.literal "0" : f32
  %1 = emitc.literal "M_PI" : f32
  %2 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<1xf32>
  %3 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<1xf32>
  %4 = "emitc.constant"() <{value = 0 : index}> : () -> index
  %5 = emitc.subscript %2[%4] : (!emitc.array<1xf32>, index) -> f32
  emitc.assign %0 : f32 to %5 : f32
  %6 = emitc.subscript %3[%4] : (!emitc.array<1xf32>, index) -> f32
  emitc.assign %1 : f32 to %6 : f32
  emitc.for %arg3 = %start to %stop step %step {
    %12 = "emitc.constant"() <{value = 0 : index}> : () -> index
    %13 = emitc.subscript %2[%12] : (!emitc.array<1xf32>, index) -> f32
    %14 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
    emitc.assign %13 : f32 to %14 : f32
    %15 = emitc.subscript %3[%12] : (!emitc.array<1xf32>, index) -> f32
    %16 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
    emitc.assign %15 : f32 to %16 : f32
    %17 = emitc.add %14, %16 : (f32, f32) -> f32
    %18 = "emitc.constant"() <{value = 0 : index}> : () -> index
    %19 = emitc.subscript %2[%18] : (!emitc.array<1xf32>, index) -> f32
    emitc.assign %17 : f32 to %19 : f32
    %20 = emitc.subscript %3[%18] : (!emitc.array<1xf32>, index) -> f32
    emitc.assign %17 : f32 to %20 : f32
  }
  %7 = "emitc.constant"() <{value = 0 : index}> : () -> index
  %8 = emitc.subscript %2[%7] : (!emitc.array<1xf32>, index) -> f32
  %9 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
  emitc.assign %8 : f32 to %9 : f32
  %10 = emitc.subscript %3[%7] : (!emitc.array<1xf32>, index) -> f32
  %11 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
  emitc.assign %10 : f32 to %11 : f32
  return
}
// CPP-DEFAULT: void test_for_yield_2() {
// CPP-DEFAULT: {{.*}}= M_PI
// CPP-DEFAULT: for (size_t [[IN:.*]] = 0; [[IN]] < 10; [[IN]] += 1) {

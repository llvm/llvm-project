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

func.func @test_for_yield() {
  %start = "emitc.constant"() <{value = 0 : index}> : () -> index
  %stop = "emitc.constant"() <{value = 10 : index}> : () -> index
  %step = "emitc.constant"() <{value = 1 : index}> : () -> index

  %s0 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
  %p0 = "emitc.constant"() <{value = 1.0 : f32}> : () -> f32

  %0 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
  %1 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<f32>
  %2 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
  %3 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<f32>
  emitc.assign %s0 : i32 to %2 : !emitc.lvalue<i32>
  emitc.assign %p0 : f32 to %3 : !emitc.lvalue<f32>
  emitc.for %iter = %start to %stop step %step {
    %4 = emitc.load %2 : !emitc.lvalue<i32>
    %sn = emitc.call_opaque "add"(%4, %iter) : (i32, index) -> i32
    %5 = emitc.load %3 : !emitc.lvalue<f32>
    %pn = emitc.call_opaque "mul"(%5, %iter) : (f32, index) -> f32
    emitc.assign %sn : i32 to %2 : !emitc.lvalue<i32>
    emitc.assign %pn : f32 to %3 : !emitc.lvalue<f32>
    emitc.yield
  }
  %6 = emitc.load %2 : !emitc.lvalue<i32>
  emitc.assign %6 : i32 to %0 : !emitc.lvalue<i32>
  %7 = emitc.load %3 : !emitc.lvalue<f32>
  emitc.assign %7 : f32 to %1 : !emitc.lvalue<f32>

  return
}
// CPP-DEFAULT: void test_for_yield() {
// CPP-DEFAULT-NEXT: size_t [[START:[^ ]*]] = 0;
// CPP-DEFAULT-NEXT: size_t [[STOP:[^ ]*]] = 10;
// CPP-DEFAULT-NEXT: size_t [[STEP:[^ ]*]] = 1;
// CPP-DEFAULT-NEXT: int32_t [[S0:[^ ]*]] = 0;
// CPP-DEFAULT-NEXT: float [[P0:[^ ]*]] = 1.000000000e+00f;
// CPP-DEFAULT-NEXT: int32_t [[SE:[^ ]*]];
// CPP-DEFAULT-NEXT: float [[PE:[^ ]*]];
// CPP-DEFAULT-NEXT: int32_t [[SI:[^ ]*]];
// CPP-DEFAULT-NEXT: float [[PI:[^ ]*]];
// CPP-DEFAULT-NEXT: [[SI:[^ ]*]] = [[S0]];
// CPP-DEFAULT-NEXT: [[PI:[^ ]*]] = [[P0]];
// CPP-DEFAULT-NEXT: for (size_t [[ITER:[^ ]*]] = [[START]]; [[ITER]] < [[STOP]]; [[ITER]] += [[STEP]]) {
// CPP-DEFAULT-NEXT: int32_t [[SI_LOAD:[^ ]*]] = [[SI]];
// CPP-DEFAULT-NEXT: int32_t [[SN:[^ ]*]] = add([[SI_LOAD]], [[ITER]]);
// CPP-DEFAULT-NEXT: float [[PI_LOAD:[^ ]*]] = [[PI]];
// CPP-DEFAULT-NEXT: float [[PN:[^ ]*]] = mul([[PI_LOAD]], [[ITER]]);
// CPP-DEFAULT-NEXT: [[SI]] = [[SN]];
// CPP-DEFAULT-NEXT: [[PI]] = [[PN]];
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-NEXT: int32_t [[SI_LOAD2:[^ ]*]] = [[SI]];
// CPP-DEFAULT-NEXT: [[SE]] = [[SI_LOAD2]];
// CPP-DEFAULT-NEXT: float [[PI_LOAD2:[^ ]*]] = [[PI]];
// CPP-DEFAULT-NEXT: [[PE]] = [[PI_LOAD2]];
// CPP-DEFAULT-NEXT: return;

// CPP-DECLTOP: void test_for_yield() {
// CPP-DECLTOP-NEXT: size_t [[START:[^ ]*]];
// CPP-DECLTOP-NEXT: size_t [[STOP:[^ ]*]];
// CPP-DECLTOP-NEXT: size_t [[STEP:[^ ]*]];
// CPP-DECLTOP-NEXT: int32_t [[S0:[^ ]*]];
// CPP-DECLTOP-NEXT: float [[P0:[^ ]*]];
// CPP-DECLTOP-NEXT: int32_t [[SE:[^ ]*]];
// CPP-DECLTOP-NEXT: float [[PE:[^ ]*]];
// CPP-DECLTOP-NEXT: int32_t [[SI:[^ ]*]];
// CPP-DECLTOP-NEXT: float [[PI:[^ ]*]];
// CPP-DECLTOP-NEXT: int32_t [[SI_LOAD:[^ ]*]];
// CPP-DECLTOP-NEXT: int32_t [[SN:[^ ]*]];
// CPP-DECLTOP-NEXT: float [[PI_LOAD:[^ ]*]];
// CPP-DECLTOP-NEXT: float [[PN:[^ ]*]];
// CPP-DECLTOP-NEXT: int32_t [[SI_LOAD2:[^ ]*]];
// CPP-DECLTOP-NEXT: float [[PI_LOAD2:[^ ]*]];
// CPP-DECLTOP-NEXT: [[START]] = 0;
// CPP-DECLTOP-NEXT: [[STOP]] = 10;
// CPP-DECLTOP-NEXT: [[STEP]] = 1;
// CPP-DECLTOP-NEXT: [[S0]] = 0;
// CPP-DECLTOP-NEXT: [[P0]] = 1.000000000e+00f;
// CPP-DECLTOP-NEXT: ;
// CPP-DECLTOP-NEXT: ;
// CPP-DECLTOP-NEXT: ;
// CPP-DECLTOP-NEXT: ;
// CPP-DECLTOP-NEXT: [[SI]] = [[S0]];
// CPP-DECLTOP-NEXT: [[PI]] = [[P0]];
// CPP-DECLTOP-NEXT: for (size_t [[ITER:[^ ]*]] = [[START]]; [[ITER]] < [[STOP]]; [[ITER]] += [[STEP]]) {
// CPP-DECLTOP-NEXT: [[SI_LOAD]] = [[SI]];
// CPP-DECLTOP-NEXT: [[SN]] = add([[SI_LOAD]], [[ITER]]);
// CPP-DECLTOP-NEXT: [[PI_LOAD]] = [[PI]];
// CPP-DECLTOP-NEXT: [[PN]] = mul([[PI_LOAD]], [[ITER]]);
// CPP-DECLTOP-NEXT: [[SI]] = [[SN]];
// CPP-DECLTOP-NEXT: [[PI]] = [[PN]];
// CPP-DECLTOP-NEXT: }
// CPP-DECLTOP-NEXT: [[SI_LOAD2]] = [[SI]];
// CPP-DECLTOP-NEXT: [[SE]] = [[SI_LOAD2]];
// CPP-DECLTOP-NEXT: [[PI_LOAD2]] = [[PI]];
// CPP-DECLTOP-NEXT: [[PE]] = [[PI_LOAD2]];
// CPP-DECLTOP-NEXT: return;

func.func @test_for_yield_2() {
  %start = emitc.literal "0" : index
  %stop = emitc.literal "10" : index
  %step = emitc.literal "1" : index

  %s0 = emitc.literal "0" : i32
  %p0 = emitc.literal "M_PI" : f32

  %0 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
  %1 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<f32>
  %2 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
  %3 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<f32>
  emitc.assign %s0 : i32 to %2 : !emitc.lvalue<i32>
  emitc.assign %p0 : f32 to %3 : !emitc.lvalue<f32>
  emitc.for %iter = %start to %stop step %step {
    %4 = emitc.load %2 : !emitc.lvalue<i32>
    %sn = emitc.call_opaque "add"(%4, %iter) : (i32, index) -> i32
    %5 = emitc.load %3 : !emitc.lvalue<f32>
    %pn = emitc.call_opaque "mul"(%5, %iter) : (f32, index) -> f32
    emitc.assign %sn : i32 to %2 : !emitc.lvalue<i32>
    emitc.assign %pn : f32 to %3 : !emitc.lvalue<f32>
    emitc.yield
  }
  %6 = emitc.load %2 : !emitc.lvalue<i32>
  emitc.assign %6 : i32 to %0 : !emitc.lvalue<i32>
  %7 = emitc.load %3 : !emitc.lvalue<f32>
  emitc.assign %7 : f32 to %1 : !emitc.lvalue<f32>

  return
}
// CPP-DEFAULT: void test_for_yield_2() {
// CPP-DEFAULT: {{.*}}= M_PI
// CPP-DEFAULT: for (size_t [[IN:.*]] = 0; [[IN]] < 10; [[IN]] += 1) {

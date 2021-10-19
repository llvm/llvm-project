// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s -check-prefix=CPP-DEFAULT
// RUN: mlir-translate -mlir-to-cpp -declare-variables-at-top %s | FileCheck %s -check-prefix=CPP-DECLTOP

func @test_for(%arg0 : index, %arg1 : index, %arg2 : index) {
  scf.for %i0 = %arg0 to %arg1 step %arg2 {
    %0 = emitc.call "f"() : () -> i32
  }
  return
}
// CPP-DEFAULT: void test_for(size_t [[START:[^ ]*]], size_t [[STOP:[^ ]*]], size_t [[STEP:[^ ]*]]) {
// CPP-DEFAULT-NEXT: for (size_t [[ITER:[^ ]*]] = [[START]]; [[ITER]] < [[STOP]]; [[ITER]] += [[STEP]]) {
// CPP-DEFAULT-NEXT: int32_t [[V4:[^ ]*]] = f();
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-NEXT: return;

// CPP-DECLTOP: void test_for(size_t [[START:[^ ]*]], size_t [[STOP:[^ ]*]], size_t [[STEP:[^ ]*]]) {
// CPP-DECLTOP-NEXT: int32_t [[V4:[^ ]*]];
// CPP-DECLTOP-NEXT: for (size_t [[ITER:[^ ]*]] = [[START]]; [[ITER]] < [[STOP]]; [[ITER]] += [[STEP]]) {
// CPP-DECLTOP-NEXT: [[V4]] = f();
// CPP-DECLTOP-NEXT: }
// CPP-DECLTOP-NEXT: return;

func @test_for_yield() {
  %start = arith.constant 0 : index
  %stop = arith.constant 10 : index
  %step = arith.constant 1 : index

  %s0 = arith.constant 0 : i32
  %p0 = arith.constant 1.0 : f32
  
  %result:2 = scf.for %iter = %start to %stop step %step iter_args(%si = %s0, %pi = %p0) -> (i32, f32) {
    %sn = emitc.call "add"(%si, %iter) : (i32, index) -> i32
    %pn = emitc.call "mul"(%pi, %iter) : (f32, index) -> f32
    scf.yield %sn, %pn : i32, f32
  }

  return
}
// CPP-DEFAULT: void test_for_yield() {
// CPP-DEFAULT-NEXT: size_t [[START:[^ ]*]] = 0;
// CPP-DEFAULT-NEXT: size_t [[STOP:[^ ]*]] = 10;
// CPP-DEFAULT-NEXT: size_t [[STEP:[^ ]*]] = 1;
// CPP-DEFAULT-NEXT: int32_t [[S0:[^ ]*]] = 0;
// CPP-DEFAULT-NEXT: float [[P0:[^ ]*]] = (float)1.000000000e+00;
// CPP-DEFAULT-NEXT: int32_t [[SE:[^ ]*]];
// CPP-DEFAULT-NEXT: float [[PE:[^ ]*]];
// CPP-DEFAULT-NEXT: int32_t [[SI:[^ ]*]] = [[S0]];
// CPP-DEFAULT-NEXT: float [[PI:[^ ]*]] = [[P0]];
// CPP-DEFAULT-NEXT: for (size_t [[ITER:[^ ]*]] = [[START]]; [[ITER]] < [[STOP]]; [[ITER]] += [[STEP]]) {
// CPP-DEFAULT-NEXT: int32_t [[SN:[^ ]*]] = add([[SI]], [[ITER]]);
// CPP-DEFAULT-NEXT: float [[PN:[^ ]*]] = mul([[PI]], [[ITER]]);
// CPP-DEFAULT-NEXT: [[SI]] = [[SN]];
// CPP-DEFAULT-NEXT: [[PI]] = [[PN]];
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-NEXT: [[SE]] = [[SI]];
// CPP-DEFAULT-NEXT: [[PE]] = [[PI]];
// CPP-DEFAULT-NEXT: return;

// CPP-DECLTOP: void test_for_yield() {
// CPP-DECLTOP-NEXT: size_t [[START:[^ ]*]];
// CPP-DECLTOP-NEXT: size_t [[STOP:[^ ]*]];
// CPP-DECLTOP-NEXT: size_t [[STEP:[^ ]*]];
// CPP-DECLTOP-NEXT: int32_t [[S0:[^ ]*]];
// CPP-DECLTOP-NEXT: float [[P0:[^ ]*]];
// CPP-DECLTOP-NEXT: int32_t [[SE:[^ ]*]];
// CPP-DECLTOP-NEXT: float [[PE:[^ ]*]];
// CPP-DECLTOP-NEXT: int32_t [[SN:[^ ]*]];
// CPP-DECLTOP-NEXT: float [[PN:[^ ]*]];
// CPP-DECLTOP-NEXT: [[START]] = 0;
// CPP-DECLTOP-NEXT: [[STOP]] = 10;
// CPP-DECLTOP-NEXT: [[STEP]] = 1;
// CPP-DECLTOP-NEXT: [[S0]] = 0;
// CPP-DECLTOP-NEXT: [[P0]] = (float)1.000000000e+00;
// CPP-DECLTOP-NEXT: int32_t [[SI:[^ ]*]] = [[S0]];
// CPP-DECLTOP-NEXT: float [[PI:[^ ]*]] = [[P0]];
// CPP-DECLTOP-NEXT: for (size_t [[ITER:[^ ]*]] = [[START]]; [[ITER]] < [[STOP]]; [[ITER]] += [[STEP]]) {
// CPP-DECLTOP-NEXT: [[SN]] = add([[SI]], [[ITER]]);
// CPP-DECLTOP-NEXT: [[PN]] = mul([[PI]], [[ITER]]);
// CPP-DECLTOP-NEXT: [[SI]] = [[SN]];
// CPP-DECLTOP-NEXT: [[PI]] = [[PN]];
// CPP-DECLTOP-NEXT: }
// CPP-DECLTOP-NEXT: [[SE]] = [[SI]];
// CPP-DECLTOP-NEXT: [[PE]] = [[PI]];
// CPP-DECLTOP-NEXT: return;

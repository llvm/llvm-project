// RUN: mlir-opt %s -convert-math-to-nvvm | FileCheck %s

// Classification ops return a bool, so their operand and result types differ.
// On shaped operands `OpToFuncCallLowering` bails out and `ScalarizeVectorOpLowering`
// unrolls them element-wise, lowering each element to a libdevice call.

// CHECK-LABEL:   func.func @fpclass_vector(
// CHECK-SAME:                              %[[ARG:.*]]: vector<2xf32>)
func.func @fpclass_vector(%arg: vector<2xf32>) -> (vector<2xi1>, vector<2xi1>, vector<2xi1>) {
  // CHECK-COUNT-2: llvm.call @__nv_isinff({{.*}}) : (f32) -> i32
  %inf = math.isinf %arg : vector<2xf32>
  // CHECK-COUNT-2: llvm.call @__nv_finitef({{.*}}) : (f32) -> i32
  %finite = math.isfinite %arg : vector<2xf32>
  // CHECK-COUNT-2: llvm.call @__nv_isnanf({{.*}}) : (f32) -> i32
  %nan = math.isnan %arg : vector<2xf32>
  return %inf, %finite, %nan : vector<2xi1>, vector<2xi1>, vector<2xi1>
}

// CHECK-LABEL:   func.func @fpclass_scalar(
func.func @fpclass_scalar(%arg: f32) -> (i1, i1, i1) {
  // CHECK: %[[INF:.*]] = llvm.call @__nv_isinff({{.*}}) : (f32) -> i32
  // CHECK: llvm.icmp "ne" %[[INF]], {{.*}} : i32
  %inf = math.isinf %arg : f32
  // CHECK: llvm.call @__nv_finitef({{.*}}) : (f32) -> i32
  %finite = math.isfinite %arg : f32
  // CHECK: llvm.call @__nv_isnanf({{.*}}) : (f32) -> i32
  %nan = math.isnan %arg : f32
  return %inf, %finite, %nan : i1, i1, i1
}

// libdevice's __nv_powif takes an i32 exponent. A narrower exponent is
// sign-extended; a wider or index exponent is not lowered.

// CHECK-LABEL:   func.func @fpowi(
func.func @fpowi(%x: f32, %e16: i16, %e32: i32, %e64: i64, %idx: index) -> (f32, f32, f32, f32) {
  // CHECK: %[[EXT:.*]] = llvm.sext %{{.*}} : i16 to i32
  // CHECK: llvm.call @__nv_powif(%{{.*}}, %[[EXT]]) : (f32, i32) -> f32
  %a = math.fpowi %x, %e16 : f32, i16
  // CHECK: llvm.call @__nv_powif(%{{.*}}, %{{.*}}) : (f32, i32) -> f32
  %b = math.fpowi %x, %e32 : f32, i32
  // CHECK: math.fpowi %{{.*}}, %{{.*}} : f32, i64
  %c = math.fpowi %x, %e64 : f32, i64
  // CHECK: math.fpowi %{{.*}}, %{{.*}} : f32, index
  %d = math.fpowi %x, %idx : f32, index
  return %a, %b, %c, %d : f32, f32, f32, f32
}

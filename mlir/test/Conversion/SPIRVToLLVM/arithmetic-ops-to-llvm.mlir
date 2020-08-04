// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.IAdd
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @iadd_scalar
func @iadd_scalar(%arg0: i32, %arg1: i32) {
  // CHECK: llvm.add %{{.*}}, %{{.*}} : !llvm.i32
  %0 = spv.IAdd %arg0, %arg1 : i32
  return
}

// CHECK-LABEL: @iadd_vector
func @iadd_vector(%arg0: vector<4xi64>, %arg1: vector<4xi64>) {
  // CHECK: llvm.add %{{.*}}, %{{.*}} : !llvm.vec<4 x i64>
  %0 = spv.IAdd %arg0, %arg1 : vector<4xi64>
  return
}

//===----------------------------------------------------------------------===//
// spv.ISub
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @isub_scalar
func @isub_scalar(%arg0: i8, %arg1: i8) {
  // CHECK: llvm.sub %{{.*}}, %{{.*}} : !llvm.i8
  %0 = spv.ISub %arg0, %arg1 : i8
  return
}

// CHECK-LABEL: @isub_vector
func @isub_vector(%arg0: vector<2xi16>, %arg1: vector<2xi16>) {
  // CHECK: llvm.sub %{{.*}}, %{{.*}} : !llvm.vec<2 x i16>
  %0 = spv.ISub %arg0, %arg1 : vector<2xi16>
  return
}

//===----------------------------------------------------------------------===//
// spv.IMul
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @imul_scalar
func @imul_scalar(%arg0: i32, %arg1: i32) {
  // CHECK: llvm.mul %{{.*}}, %{{.*}} : !llvm.i32
  %0 = spv.IMul %arg0, %arg1 : i32
  return
}

// CHECK-LABEL: @imul_vector
func @imul_vector(%arg0: vector<3xi32>, %arg1: vector<3xi32>) {
  // CHECK: llvm.mul %{{.*}}, %{{.*}} : !llvm.vec<3 x i32>
  %0 = spv.IMul %arg0, %arg1 : vector<3xi32>
  return
}

//===----------------------------------------------------------------------===//
// spv.FAdd
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fadd_scalar
func @fadd_scalar(%arg0: f16, %arg1: f16) {
  // CHECK: llvm.fadd %{{.*}}, %{{.*}} : !llvm.half
  %0 = spv.FAdd %arg0, %arg1 : f16
  return
}

// CHECK-LABEL: @fadd_vector
func @fadd_vector(%arg0: vector<4xf32>, %arg1: vector<4xf32>) {
  // CHECK: llvm.fadd %{{.*}}, %{{.*}} : !llvm.vec<4 x float>
  %0 = spv.FAdd %arg0, %arg1 : vector<4xf32>
  return
}

//===----------------------------------------------------------------------===//
// spv.FSub
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fsub_scalar
func @fsub_scalar(%arg0: f32, %arg1: f32) {
  // CHECK: llvm.fsub %{{.*}}, %{{.*}} : !llvm.float
  %0 = spv.FSub %arg0, %arg1 : f32
  return
}

// CHECK-LABEL: @fsub_vector
func @fsub_vector(%arg0: vector<2xf32>, %arg1: vector<2xf32>) {
  // CHECK: llvm.fsub %{{.*}}, %{{.*}} : !llvm.vec<2 x float>
  %0 = spv.FSub %arg0, %arg1 : vector<2xf32>
  return
}

//===----------------------------------------------------------------------===//
// spv.FDiv
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fdiv_scalar
func @fdiv_scalar(%arg0: f32, %arg1: f32) {
  // CHECK: llvm.fdiv %{{.*}}, %{{.*}} : !llvm.float
  %0 = spv.FDiv %arg0, %arg1 : f32
  return
}

// CHECK-LABEL: @fdiv_vector
func @fdiv_vector(%arg0: vector<3xf64>, %arg1: vector<3xf64>) {
  // CHECK: llvm.fdiv %{{.*}}, %{{.*}} : !llvm.vec<3 x double>
  %0 = spv.FDiv %arg0, %arg1 : vector<3xf64>
  return
}

//===----------------------------------------------------------------------===//
// spv.FMul
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fmul_scalar
func @fmul_scalar(%arg0: f32, %arg1: f32) {
  // CHECK: llvm.fmul %{{.*}}, %{{.*}} : !llvm.float
  %0 = spv.FMul %arg0, %arg1 : f32
  return
}

// CHECK-LABEL: @fmul_vector
func @fmul_vector(%arg0: vector<2xf32>, %arg1: vector<2xf32>) {
  // CHECK: llvm.fmul %{{.*}}, %{{.*}} : !llvm.vec<2 x float>
  %0 = spv.FMul %arg0, %arg1 : vector<2xf32>
  return
}

//===----------------------------------------------------------------------===//
// spv.FRem
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @frem_scalar
func @frem_scalar(%arg0: f32, %arg1: f32) {
  // CHECK: llvm.frem %{{.*}}, %{{.*}} : !llvm.float
  %0 = spv.FRem %arg0, %arg1 : f32
  return
}

// CHECK-LABEL: @frem_vector
func @frem_vector(%arg0: vector<3xf64>, %arg1: vector<3xf64>) {
  // CHECK: llvm.frem %{{.*}}, %{{.*}} : !llvm.vec<3 x double>
  %0 = spv.FRem %arg0, %arg1 : vector<3xf64>
  return
}

//===----------------------------------------------------------------------===//
// spv.FNegate
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fneg_scalar
func @fneg_scalar(%arg: f64) {
  // CHECK: llvm.fneg %{{.*}} : !llvm.double
  %0 = spv.FNegate %arg : f64
  return
}

// CHECK-LABEL: @fneg_vector
func @fneg_vector(%arg: vector<2xf32>) {
  // CHECK: llvm.fneg %{{.*}} : !llvm.vec<2 x float>
  %0 = spv.FNegate %arg : vector<2xf32>
  return
}

//===----------------------------------------------------------------------===//
// spv.UDiv
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @udiv_scalar
func @udiv_scalar(%arg0: i32, %arg1: i32) {
  // CHECK: llvm.udiv %{{.*}}, %{{.*}} : !llvm.i32
  %0 = spv.UDiv %arg0, %arg1 : i32
  return
}

// CHECK-LABEL: @udiv_vector
func @udiv_vector(%arg0: vector<3xi64>, %arg1: vector<3xi64>) {
  // CHECK: llvm.udiv %{{.*}}, %{{.*}} : !llvm.vec<3 x i64>
  %0 = spv.UDiv %arg0, %arg1 : vector<3xi64>
  return
}

//===----------------------------------------------------------------------===//
// spv.UMod
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @umod_scalar
func @umod_scalar(%arg0: i32, %arg1: i32) {
  // CHECK: llvm.urem %{{.*}}, %{{.*}} : !llvm.i32
  %0 = spv.UMod %arg0, %arg1 : i32
  return
}

// CHECK-LABEL: @umod_vector
func @umod_vector(%arg0: vector<3xi64>, %arg1: vector<3xi64>) {
  // CHECK: llvm.urem %{{.*}}, %{{.*}} : !llvm.vec<3 x i64>
  %0 = spv.UMod %arg0, %arg1 : vector<3xi64>
  return
}

//===----------------------------------------------------------------------===//
// spv.SDiv
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @sdiv_scalar
func @sdiv_scalar(%arg0: i16, %arg1: i16) {
  // CHECK: llvm.sdiv %{{.*}}, %{{.*}} : !llvm.i16
  %0 = spv.SDiv %arg0, %arg1 : i16
  return
}

// CHECK-LABEL: @sdiv_vector
func @sdiv_vector(%arg0: vector<2xi64>, %arg1: vector<2xi64>) {
  // CHECK: llvm.sdiv %{{.*}}, %{{.*}} : !llvm.vec<2 x i64>
  %0 = spv.SDiv %arg0, %arg1 : vector<2xi64>
  return
}

//===----------------------------------------------------------------------===//
// spv.SRem
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @srem_scalar
func @srem_scalar(%arg0: i32, %arg1: i32) {
  // CHECK: llvm.srem %{{.*}}, %{{.*}} : !llvm.i32
  %0 = spv.SRem %arg0, %arg1 : i32
  return
}

// CHECK-LABEL: @srem_vector
func @srem_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) {
  // CHECK: llvm.srem %{{.*}}, %{{.*}} : !llvm.vec<4 x i32>
  %0 = spv.SRem %arg0, %arg1 : vector<4xi32>
  return
}

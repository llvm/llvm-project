// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.IAdd
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @iadd_scalar
spv.func @iadd_scalar(%arg0: i32, %arg1: i32) "None" {
  // CHECK: llvm.add %{{.*}}, %{{.*}} : i32
  %0 = spv.IAdd %arg0, %arg1 : i32
  spv.Return
}

// CHECK-LABEL: @iadd_vector
spv.func @iadd_vector(%arg0: vector<4xi64>, %arg1: vector<4xi64>) "None" {
  // CHECK: llvm.add %{{.*}}, %{{.*}} : vector<4xi64>
  %0 = spv.IAdd %arg0, %arg1 : vector<4xi64>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.ISub
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @isub_scalar
spv.func @isub_scalar(%arg0: i8, %arg1: i8) "None" {
  // CHECK: llvm.sub %{{.*}}, %{{.*}} : i8
  %0 = spv.ISub %arg0, %arg1 : i8
  spv.Return
}

// CHECK-LABEL: @isub_vector
spv.func @isub_vector(%arg0: vector<2xi16>, %arg1: vector<2xi16>) "None" {
  // CHECK: llvm.sub %{{.*}}, %{{.*}} : vector<2xi16>
  %0 = spv.ISub %arg0, %arg1 : vector<2xi16>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.IMul
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @imul_scalar
spv.func @imul_scalar(%arg0: i32, %arg1: i32) "None" {
  // CHECK: llvm.mul %{{.*}}, %{{.*}} : i32
  %0 = spv.IMul %arg0, %arg1 : i32
  spv.Return
}

// CHECK-LABEL: @imul_vector
spv.func @imul_vector(%arg0: vector<3xi32>, %arg1: vector<3xi32>) "None" {
  // CHECK: llvm.mul %{{.*}}, %{{.*}} : vector<3xi32>
  %0 = spv.IMul %arg0, %arg1 : vector<3xi32>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.FAdd
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fadd_scalar
spv.func @fadd_scalar(%arg0: f16, %arg1: f16) "None" {
  // CHECK: llvm.fadd %{{.*}}, %{{.*}} : f16
  %0 = spv.FAdd %arg0, %arg1 : f16
  spv.Return
}

// CHECK-LABEL: @fadd_vector
spv.func @fadd_vector(%arg0: vector<4xf32>, %arg1: vector<4xf32>) "None" {
  // CHECK: llvm.fadd %{{.*}}, %{{.*}} : vector<4xf32>
  %0 = spv.FAdd %arg0, %arg1 : vector<4xf32>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.FSub
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fsub_scalar
spv.func @fsub_scalar(%arg0: f32, %arg1: f32) "None" {
  // CHECK: llvm.fsub %{{.*}}, %{{.*}} : f32
  %0 = spv.FSub %arg0, %arg1 : f32
  spv.Return
}

// CHECK-LABEL: @fsub_vector
spv.func @fsub_vector(%arg0: vector<2xf32>, %arg1: vector<2xf32>) "None" {
  // CHECK: llvm.fsub %{{.*}}, %{{.*}} : vector<2xf32>
  %0 = spv.FSub %arg0, %arg1 : vector<2xf32>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.FDiv
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fdiv_scalar
spv.func @fdiv_scalar(%arg0: f32, %arg1: f32) "None" {
  // CHECK: llvm.fdiv %{{.*}}, %{{.*}} : f32
  %0 = spv.FDiv %arg0, %arg1 : f32
  spv.Return
}

// CHECK-LABEL: @fdiv_vector
spv.func @fdiv_vector(%arg0: vector<3xf64>, %arg1: vector<3xf64>) "None" {
  // CHECK: llvm.fdiv %{{.*}}, %{{.*}} : vector<3xf64>
  %0 = spv.FDiv %arg0, %arg1 : vector<3xf64>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.FMul
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fmul_scalar
spv.func @fmul_scalar(%arg0: f32, %arg1: f32) "None" {
  // CHECK: llvm.fmul %{{.*}}, %{{.*}} : f32
  %0 = spv.FMul %arg0, %arg1 : f32
  spv.Return
}

// CHECK-LABEL: @fmul_vector
spv.func @fmul_vector(%arg0: vector<2xf32>, %arg1: vector<2xf32>) "None" {
  // CHECK: llvm.fmul %{{.*}}, %{{.*}} : vector<2xf32>
  %0 = spv.FMul %arg0, %arg1 : vector<2xf32>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.FRem
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @frem_scalar
spv.func @frem_scalar(%arg0: f32, %arg1: f32) "None" {
  // CHECK: llvm.frem %{{.*}}, %{{.*}} : f32
  %0 = spv.FRem %arg0, %arg1 : f32
  spv.Return
}

// CHECK-LABEL: @frem_vector
spv.func @frem_vector(%arg0: vector<3xf64>, %arg1: vector<3xf64>) "None" {
  // CHECK: llvm.frem %{{.*}}, %{{.*}} : vector<3xf64>
  %0 = spv.FRem %arg0, %arg1 : vector<3xf64>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.FNegate
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fneg_scalar
spv.func @fneg_scalar(%arg: f64) "None" {
  // CHECK: llvm.fneg %{{.*}} : f64
  %0 = spv.FNegate %arg : f64
  spv.Return
}

// CHECK-LABEL: @fneg_vector
spv.func @fneg_vector(%arg: vector<2xf32>) "None" {
  // CHECK: llvm.fneg %{{.*}} : vector<2xf32>
  %0 = spv.FNegate %arg : vector<2xf32>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.UDiv
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @udiv_scalar
spv.func @udiv_scalar(%arg0: i32, %arg1: i32) "None" {
  // CHECK: llvm.udiv %{{.*}}, %{{.*}} : i32
  %0 = spv.UDiv %arg0, %arg1 : i32
  spv.Return
}

// CHECK-LABEL: @udiv_vector
spv.func @udiv_vector(%arg0: vector<3xi64>, %arg1: vector<3xi64>) "None" {
  // CHECK: llvm.udiv %{{.*}}, %{{.*}} : vector<3xi64>
  %0 = spv.UDiv %arg0, %arg1 : vector<3xi64>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.UMod
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @umod_scalar
spv.func @umod_scalar(%arg0: i32, %arg1: i32) "None" {
  // CHECK: llvm.urem %{{.*}}, %{{.*}} : i32
  %0 = spv.UMod %arg0, %arg1 : i32
  spv.Return
}

// CHECK-LABEL: @umod_vector
spv.func @umod_vector(%arg0: vector<3xi64>, %arg1: vector<3xi64>) "None" {
  // CHECK: llvm.urem %{{.*}}, %{{.*}} : vector<3xi64>
  %0 = spv.UMod %arg0, %arg1 : vector<3xi64>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.SDiv
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @sdiv_scalar
spv.func @sdiv_scalar(%arg0: i16, %arg1: i16) "None" {
  // CHECK: llvm.sdiv %{{.*}}, %{{.*}} : i16
  %0 = spv.SDiv %arg0, %arg1 : i16
  spv.Return
}

// CHECK-LABEL: @sdiv_vector
spv.func @sdiv_vector(%arg0: vector<2xi64>, %arg1: vector<2xi64>) "None" {
  // CHECK: llvm.sdiv %{{.*}}, %{{.*}} : vector<2xi64>
  %0 = spv.SDiv %arg0, %arg1 : vector<2xi64>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.SRem
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @srem_scalar
spv.func @srem_scalar(%arg0: i32, %arg1: i32) "None" {
  // CHECK: llvm.srem %{{.*}}, %{{.*}} : i32
  %0 = spv.SRem %arg0, %arg1 : i32
  spv.Return
}

// CHECK-LABEL: @srem_vector
spv.func @srem_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) "None" {
  // CHECK: llvm.srem %{{.*}}, %{{.*}} : vector<4xi32>
  %0 = spv.SRem %arg0, %arg1 : vector<4xi32>
  spv.Return
}

// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.Bitcast
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @bitcast_float_to_integer_scalar
func @bitcast_float_to_integer_scalar(%arg0 : f32) {
  // CHECK: llvm.bitcast {{.*}} : !llvm.float to !llvm.i32
  %0 = spv.Bitcast %arg0: f32 to i32
  return
}

// CHECK-LABEL: @bitcast_float_to_integer_vector
func @bitcast_float_to_integer_vector(%arg0 : vector<3xf32>) {
  // CHECK: {{.*}} = llvm.bitcast {{.*}} : !llvm.vec<3 x float> to !llvm.vec<3 x i32>
  %0 = spv.Bitcast %arg0: vector<3xf32> to vector<3xi32>
  return
}

// CHECK-LABEL: @bitcast_vector_to_scalar
func @bitcast_vector_to_scalar(%arg0 : vector<2xf32>) {
  // CHECK: {{.*}} = llvm.bitcast {{.*}} : !llvm.vec<2 x float> to !llvm.i64
  %0 = spv.Bitcast %arg0: vector<2xf32> to i64
  return
}

// CHECK-LABEL: @bitcast_scalar_to_vector
func @bitcast_scalar_to_vector(%arg0 : f64) {
  // CHECK: {{.*}} = llvm.bitcast {{.*}} : !llvm.double to !llvm.vec<2 x i32>
  %0 = spv.Bitcast %arg0: f64 to vector<2xi32>
  return
}

// CHECK-LABEL: @bitcast_vector_to_vector
func @bitcast_vector_to_vector(%arg0 : vector<4xf32>) {
  // CHECK: {{.*}} = llvm.bitcast {{.*}} : !llvm.vec<4 x float> to !llvm.vec<2 x i64>
  %0 = spv.Bitcast %arg0: vector<4xf32> to vector<2xi64>
  return
}

// CHECK-LABEL: @bitcast_pointer
func @bitcast_pointer(%arg0: !spv.ptr<f32, Function>) {
  // CHECK: llvm.bitcast %{{.*}} : !llvm.ptr<float> to !llvm.ptr<i32>
  %0 = spv.Bitcast %arg0 : !spv.ptr<f32, Function> to !spv.ptr<i32, Function>
  return
}

//===----------------------------------------------------------------------===//
// spv.ConvertFToS
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @convert_float_to_signed_scalar
func @convert_float_to_signed_scalar(%arg0: f32) {
  // CHECK: llvm.fptosi %{{.*}} : !llvm.float to !llvm.i32
  %0 = spv.ConvertFToS %arg0: f32 to i32
  return
}

// CHECK-LABEL: @convert_float_to_signed_vector
func @convert_float_to_signed_vector(%arg0: vector<2xf32>) {
  // CHECK: llvm.fptosi %{{.*}} : !llvm.vec<2 x float> to !llvm.vec<2 x i32>
    %0 = spv.ConvertFToS %arg0: vector<2xf32> to vector<2xi32>
  return
}

//===----------------------------------------------------------------------===//
// spv.ConvertFToU
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @convert_float_to_unsigned_scalar
func @convert_float_to_unsigned_scalar(%arg0: f32) {
  // CHECK: llvm.fptoui %{{.*}} : !llvm.float to !llvm.i32
  %0 = spv.ConvertFToU %arg0: f32 to i32
  return
}

// CHECK-LABEL: @convert_float_to_unsigned_vector
func @convert_float_to_unsigned_vector(%arg0: vector<2xf32>) {
  // CHECK: llvm.fptoui %{{.*}} : !llvm.vec<2 x float> to !llvm.vec<2 x i32>
    %0 = spv.ConvertFToU %arg0: vector<2xf32> to vector<2xi32>
  return
}

//===----------------------------------------------------------------------===//
// spv.ConvertSToF
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @convert_signed_to_float_scalar
func @convert_signed_to_float_scalar(%arg0: i32) {
  // CHECK: llvm.sitofp %{{.*}} : !llvm.i32 to !llvm.float
  %0 = spv.ConvertSToF %arg0: i32 to f32
  return
}

// CHECK-LABEL: @convert_signed_to_float_vector
func @convert_signed_to_float_vector(%arg0: vector<3xi32>) {
  // CHECK: llvm.sitofp %{{.*}} : !llvm.vec<3 x i32> to !llvm.vec<3 x float>
    %0 = spv.ConvertSToF %arg0: vector<3xi32> to vector<3xf32>
  return
}

//===----------------------------------------------------------------------===//
// spv.ConvertUToF
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @convert_unsigned_to_float_scalar
func @convert_unsigned_to_float_scalar(%arg0: i32) {
  // CHECK: llvm.uitofp %{{.*}} : !llvm.i32 to !llvm.float
  %0 = spv.ConvertUToF %arg0: i32 to f32
  return
}

// CHECK-LABEL: @convert_unsigned_to_float_vector
func @convert_unsigned_to_float_vector(%arg0: vector<3xi32>) {
  // CHECK: llvm.uitofp %{{.*}} : !llvm.vec<3 x i32> to !llvm.vec<3 x float>
    %0 = spv.ConvertUToF %arg0: vector<3xi32> to vector<3xf32>
  return
}

//===----------------------------------------------------------------------===//
// spv.FConvert
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fconvert_scalar
func @fconvert_scalar(%arg0: f32, %arg1: f64) {
  // CHECK: llvm.fpext %{{.*}} : !llvm.float to !llvm.double
  %0 = spv.FConvert %arg0: f32 to f64

  // CHECK: llvm.fptrunc %{{.*}} : !llvm.double to !llvm.float
  %1 = spv.FConvert %arg1: f64 to f32
  return
}

// CHECK-LABEL: @fconvert_vector
func @fconvert_vector(%arg0: vector<2xf32>, %arg1: vector<2xf64>) {
  // CHECK: llvm.fpext %{{.*}} : !llvm.vec<2 x float> to !llvm.vec<2 x double>
  %0 = spv.FConvert %arg0: vector<2xf32> to vector<2xf64>

  // CHECK: llvm.fptrunc %{{.*}} : !llvm.vec<2 x double> to !llvm.vec<2 x float>
  %1 = spv.FConvert %arg1: vector<2xf64> to vector<2xf32>
  return
}

//===----------------------------------------------------------------------===//
// spv.SConvert
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @sconvert_scalar
func @sconvert_scalar(%arg0: i32, %arg1: i64) {
  // CHECK: llvm.sext %{{.*}} : !llvm.i32 to !llvm.i64
  %0 = spv.SConvert %arg0: i32 to i64

  // CHECK: llvm.trunc %{{.*}} : !llvm.i64 to !llvm.i32
  %1 = spv.SConvert %arg1: i64 to i32
  return
}

// CHECK-LABEL: @sconvert_vector
func @sconvert_vector(%arg0: vector<3xi32>, %arg1: vector<3xi64>) {
  // CHECK: llvm.sext %{{.*}} : !llvm.vec<3 x i32> to !llvm.vec<3 x i64>
  %0 = spv.SConvert %arg0: vector<3xi32> to vector<3xi64>

  // CHECK: llvm.trunc %{{.*}} : !llvm.vec<3 x i64> to !llvm.vec<3 x i32>
  %1 = spv.SConvert %arg1: vector<3xi64> to vector<3xi32>
  return
}

//===----------------------------------------------------------------------===//
// spv.UConvert
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @uconvert_scalar
func @uconvert_scalar(%arg0: i32, %arg1: i64) {
  // CHECK: llvm.zext %{{.*}} : !llvm.i32 to !llvm.i64
  %0 = spv.UConvert %arg0: i32 to i64

  // CHECK: llvm.trunc %{{.*}} : !llvm.i64 to !llvm.i32
  %1 = spv.UConvert %arg1: i64 to i32
  return
}

// CHECK-LABEL: @uconvert_vector
func @uconvert_vector(%arg0: vector<3xi32>, %arg1: vector<3xi64>) {
  // CHECK: llvm.zext %{{.*}} : !llvm.vec<3 x i32> to !llvm.vec<3 x i64>
  %0 = spv.UConvert %arg0: vector<3xi32> to vector<3xi64>

  // CHECK: llvm.trunc %{{.*}} : !llvm.vec<3 x i64> to !llvm.vec<3 x i32>
  %1 = spv.UConvert %arg1: vector<3xi64> to vector<3xi32>
  return
}

// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.Bitcast
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @bitcast_float_to_integer_scalar
spv.func @bitcast_float_to_integer_scalar(%arg0 : f32) "None" {
  // CHECK: llvm.bitcast {{.*}} : !llvm.float to !llvm.i32
  %0 = spv.Bitcast %arg0: f32 to i32
  spv.Return
}

// CHECK-LABEL: @bitcast_float_to_integer_vector
spv.func @bitcast_float_to_integer_vector(%arg0 : vector<3xf32>) "None" {
  // CHECK: {{.*}} = llvm.bitcast {{.*}} : !llvm.vec<3 x float> to !llvm.vec<3 x i32>
  %0 = spv.Bitcast %arg0: vector<3xf32> to vector<3xi32>
  spv.Return
}

// CHECK-LABEL: @bitcast_vector_to_scalar
spv.func @bitcast_vector_to_scalar(%arg0 : vector<2xf32>) "None" {
  // CHECK: {{.*}} = llvm.bitcast {{.*}} : !llvm.vec<2 x float> to !llvm.i64
  %0 = spv.Bitcast %arg0: vector<2xf32> to i64
  spv.Return
}

// CHECK-LABEL: @bitcast_scalar_to_vector
spv.func @bitcast_scalar_to_vector(%arg0 : f64) "None" {
  // CHECK: {{.*}} = llvm.bitcast {{.*}} : !llvm.double to !llvm.vec<2 x i32>
  %0 = spv.Bitcast %arg0: f64 to vector<2xi32>
  spv.Return
}

// CHECK-LABEL: @bitcast_vector_to_vector
spv.func @bitcast_vector_to_vector(%arg0 : vector<4xf32>) "None" {
  // CHECK: {{.*}} = llvm.bitcast {{.*}} : !llvm.vec<4 x float> to !llvm.vec<2 x i64>
  %0 = spv.Bitcast %arg0: vector<4xf32> to vector<2xi64>
  spv.Return
}

// CHECK-LABEL: @bitcast_pointer
spv.func @bitcast_pointer(%arg0: !spv.ptr<f32, Function>) "None" {
  // CHECK: llvm.bitcast %{{.*}} : !llvm.ptr<float> to !llvm.ptr<i32>
  %0 = spv.Bitcast %arg0 : !spv.ptr<f32, Function> to !spv.ptr<i32, Function>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.ConvertFToS
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @convert_float_to_signed_scalar
spv.func @convert_float_to_signed_scalar(%arg0: f32) "None" {
  // CHECK: llvm.fptosi %{{.*}} : !llvm.float to !llvm.i32
  %0 = spv.ConvertFToS %arg0: f32 to i32
  spv.Return
}

// CHECK-LABEL: @convert_float_to_signed_vector
spv.func @convert_float_to_signed_vector(%arg0: vector<2xf32>) "None" {
  // CHECK: llvm.fptosi %{{.*}} : !llvm.vec<2 x float> to !llvm.vec<2 x i32>
    %0 = spv.ConvertFToS %arg0: vector<2xf32> to vector<2xi32>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.ConvertFToU
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @convert_float_to_unsigned_scalar
spv.func @convert_float_to_unsigned_scalar(%arg0: f32) "None" {
  // CHECK: llvm.fptoui %{{.*}} : !llvm.float to !llvm.i32
  %0 = spv.ConvertFToU %arg0: f32 to i32
  spv.Return
}

// CHECK-LABEL: @convert_float_to_unsigned_vector
spv.func @convert_float_to_unsigned_vector(%arg0: vector<2xf32>) "None" {
  // CHECK: llvm.fptoui %{{.*}} : !llvm.vec<2 x float> to !llvm.vec<2 x i32>
    %0 = spv.ConvertFToU %arg0: vector<2xf32> to vector<2xi32>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.ConvertSToF
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @convert_signed_to_float_scalar
spv.func @convert_signed_to_float_scalar(%arg0: i32) "None" {
  // CHECK: llvm.sitofp %{{.*}} : !llvm.i32 to !llvm.float
  %0 = spv.ConvertSToF %arg0: i32 to f32
  spv.Return
}

// CHECK-LABEL: @convert_signed_to_float_vector
spv.func @convert_signed_to_float_vector(%arg0: vector<3xi32>) "None" {
  // CHECK: llvm.sitofp %{{.*}} : !llvm.vec<3 x i32> to !llvm.vec<3 x float>
    %0 = spv.ConvertSToF %arg0: vector<3xi32> to vector<3xf32>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.ConvertUToF
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @convert_unsigned_to_float_scalar
spv.func @convert_unsigned_to_float_scalar(%arg0: i32) "None" {
  // CHECK: llvm.uitofp %{{.*}} : !llvm.i32 to !llvm.float
  %0 = spv.ConvertUToF %arg0: i32 to f32
  spv.Return
}

// CHECK-LABEL: @convert_unsigned_to_float_vector
spv.func @convert_unsigned_to_float_vector(%arg0: vector<3xi32>) "None" {
  // CHECK: llvm.uitofp %{{.*}} : !llvm.vec<3 x i32> to !llvm.vec<3 x float>
    %0 = spv.ConvertUToF %arg0: vector<3xi32> to vector<3xf32>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.FConvert
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fconvert_scalar
spv.func @fconvert_scalar(%arg0: f32, %arg1: f64) "None" {
  // CHECK: llvm.fpext %{{.*}} : !llvm.float to !llvm.double
  %0 = spv.FConvert %arg0: f32 to f64

  // CHECK: llvm.fptrunc %{{.*}} : !llvm.double to !llvm.float
  %1 = spv.FConvert %arg1: f64 to f32
  spv.Return
}

// CHECK-LABEL: @fconvert_vector
spv.func @fconvert_vector(%arg0: vector<2xf32>, %arg1: vector<2xf64>) "None" {
  // CHECK: llvm.fpext %{{.*}} : !llvm.vec<2 x float> to !llvm.vec<2 x double>
  %0 = spv.FConvert %arg0: vector<2xf32> to vector<2xf64>

  // CHECK: llvm.fptrunc %{{.*}} : !llvm.vec<2 x double> to !llvm.vec<2 x float>
  %1 = spv.FConvert %arg1: vector<2xf64> to vector<2xf32>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.SConvert
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @sconvert_scalar
spv.func @sconvert_scalar(%arg0: i32, %arg1: i64) "None" {
  // CHECK: llvm.sext %{{.*}} : !llvm.i32 to !llvm.i64
  %0 = spv.SConvert %arg0: i32 to i64

  // CHECK: llvm.trunc %{{.*}} : !llvm.i64 to !llvm.i32
  %1 = spv.SConvert %arg1: i64 to i32
  spv.Return
}

// CHECK-LABEL: @sconvert_vector
spv.func @sconvert_vector(%arg0: vector<3xi32>, %arg1: vector<3xi64>) "None" {
  // CHECK: llvm.sext %{{.*}} : !llvm.vec<3 x i32> to !llvm.vec<3 x i64>
  %0 = spv.SConvert %arg0: vector<3xi32> to vector<3xi64>

  // CHECK: llvm.trunc %{{.*}} : !llvm.vec<3 x i64> to !llvm.vec<3 x i32>
  %1 = spv.SConvert %arg1: vector<3xi64> to vector<3xi32>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.UConvert
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @uconvert_scalar
spv.func @uconvert_scalar(%arg0: i32, %arg1: i64) "None" {
  // CHECK: llvm.zext %{{.*}} : !llvm.i32 to !llvm.i64
  %0 = spv.UConvert %arg0: i32 to i64

  // CHECK: llvm.trunc %{{.*}} : !llvm.i64 to !llvm.i32
  %1 = spv.UConvert %arg1: i64 to i32
  spv.Return
}

// CHECK-LABEL: @uconvert_vector
spv.func @uconvert_vector(%arg0: vector<3xi32>, %arg1: vector<3xi64>) "None" {
  // CHECK: llvm.zext %{{.*}} : !llvm.vec<3 x i32> to !llvm.vec<3 x i64>
  %0 = spv.UConvert %arg0: vector<3xi32> to vector<3xi64>

  // CHECK: llvm.trunc %{{.*}} : !llvm.vec<3 x i64> to !llvm.vec<3 x i32>
  %1 = spv.UConvert %arg1: vector<3xi64> to vector<3xi32>
  spv.Return
}

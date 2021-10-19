// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.Bitcast
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @bitcast_float_to_integer_scalar
spv.func @bitcast_float_to_integer_scalar(%arg0 : f32) "None" {
  // CHECK: llvm.bitcast {{.*}} : f32 to i32
  %0 = spv.Bitcast %arg0: f32 to i32
  spv.Return
}

// CHECK-LABEL: @bitcast_float_to_integer_vector
spv.func @bitcast_float_to_integer_vector(%arg0 : vector<3xf32>) "None" {
  // CHECK: {{.*}} = llvm.bitcast {{.*}} : vector<3xf32> to vector<3xi32>
  %0 = spv.Bitcast %arg0: vector<3xf32> to vector<3xi32>
  spv.Return
}

// CHECK-LABEL: @bitcast_vector_to_scalar
spv.func @bitcast_vector_to_scalar(%arg0 : vector<2xf32>) "None" {
  // CHECK: {{.*}} = llvm.bitcast {{.*}} : vector<2xf32> to i64
  %0 = spv.Bitcast %arg0: vector<2xf32> to i64
  spv.Return
}

// CHECK-LABEL: @bitcast_scalar_to_vector
spv.func @bitcast_scalar_to_vector(%arg0 : f64) "None" {
  // CHECK: {{.*}} = llvm.bitcast {{.*}} : f64 to vector<2xi32>
  %0 = spv.Bitcast %arg0: f64 to vector<2xi32>
  spv.Return
}

// CHECK-LABEL: @bitcast_vector_to_vector
spv.func @bitcast_vector_to_vector(%arg0 : vector<4xf32>) "None" {
  // CHECK: {{.*}} = llvm.bitcast {{.*}} : vector<4xf32> to vector<2xi64>
  %0 = spv.Bitcast %arg0: vector<4xf32> to vector<2xi64>
  spv.Return
}

// CHECK-LABEL: @bitcast_pointer
spv.func @bitcast_pointer(%arg0: !spv.ptr<f32, Function>) "None" {
  // CHECK: llvm.bitcast %{{.*}} : !llvm.ptr<f32> to !llvm.ptr<i32>
  %0 = spv.Bitcast %arg0 : !spv.ptr<f32, Function> to !spv.ptr<i32, Function>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.ConvertFToS
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @convert_float_to_signed_scalar
spv.func @convert_float_to_signed_scalar(%arg0: f32) "None" {
  // CHECK: llvm.fptosi %{{.*}} : f32 to i32
  %0 = spv.ConvertFToS %arg0: f32 to i32
  spv.Return
}

// CHECK-LABEL: @convert_float_to_signed_vector
spv.func @convert_float_to_signed_vector(%arg0: vector<2xf32>) "None" {
  // CHECK: llvm.fptosi %{{.*}} : vector<2xf32> to vector<2xi32>
    %0 = spv.ConvertFToS %arg0: vector<2xf32> to vector<2xi32>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.ConvertFToU
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @convert_float_to_unsigned_scalar
spv.func @convert_float_to_unsigned_scalar(%arg0: f32) "None" {
  // CHECK: llvm.fptoui %{{.*}} : f32 to i32
  %0 = spv.ConvertFToU %arg0: f32 to i32
  spv.Return
}

// CHECK-LABEL: @convert_float_to_unsigned_vector
spv.func @convert_float_to_unsigned_vector(%arg0: vector<2xf32>) "None" {
  // CHECK: llvm.fptoui %{{.*}} : vector<2xf32> to vector<2xi32>
    %0 = spv.ConvertFToU %arg0: vector<2xf32> to vector<2xi32>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.ConvertSToF
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @convert_signed_to_float_scalar
spv.func @convert_signed_to_float_scalar(%arg0: i32) "None" {
  // CHECK: llvm.sitofp %{{.*}} : i32 to f32
  %0 = spv.ConvertSToF %arg0: i32 to f32
  spv.Return
}

// CHECK-LABEL: @convert_signed_to_float_vector
spv.func @convert_signed_to_float_vector(%arg0: vector<3xi32>) "None" {
  // CHECK: llvm.sitofp %{{.*}} : vector<3xi32> to vector<3xf32>
    %0 = spv.ConvertSToF %arg0: vector<3xi32> to vector<3xf32>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.ConvertUToF
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @convert_unsigned_to_float_scalar
spv.func @convert_unsigned_to_float_scalar(%arg0: i32) "None" {
  // CHECK: llvm.uitofp %{{.*}} : i32 to f32
  %0 = spv.ConvertUToF %arg0: i32 to f32
  spv.Return
}

// CHECK-LABEL: @convert_unsigned_to_float_vector
spv.func @convert_unsigned_to_float_vector(%arg0: vector<3xi32>) "None" {
  // CHECK: llvm.uitofp %{{.*}} : vector<3xi32> to vector<3xf32>
    %0 = spv.ConvertUToF %arg0: vector<3xi32> to vector<3xf32>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.FConvert
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fconvert_scalar
spv.func @fconvert_scalar(%arg0: f32, %arg1: f64) "None" {
  // CHECK: llvm.fpext %{{.*}} : f32 to f64
  %0 = spv.FConvert %arg0: f32 to f64

  // CHECK: llvm.fptrunc %{{.*}} : f64 to f32
  %1 = spv.FConvert %arg1: f64 to f32
  spv.Return
}

// CHECK-LABEL: @fconvert_vector
spv.func @fconvert_vector(%arg0: vector<2xf32>, %arg1: vector<2xf64>) "None" {
  // CHECK: llvm.fpext %{{.*}} : vector<2xf32> to vector<2xf64>
  %0 = spv.FConvert %arg0: vector<2xf32> to vector<2xf64>

  // CHECK: llvm.fptrunc %{{.*}} : vector<2xf64> to vector<2xf32>
  %1 = spv.FConvert %arg1: vector<2xf64> to vector<2xf32>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.SConvert
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @sconvert_scalar
spv.func @sconvert_scalar(%arg0: i32, %arg1: i64) "None" {
  // CHECK: llvm.sext %{{.*}} : i32 to i64
  %0 = spv.SConvert %arg0: i32 to i64

  // CHECK: llvm.trunc %{{.*}} : i64 to i32
  %1 = spv.SConvert %arg1: i64 to i32
  spv.Return
}

// CHECK-LABEL: @sconvert_vector
spv.func @sconvert_vector(%arg0: vector<3xi32>, %arg1: vector<3xi64>) "None" {
  // CHECK: llvm.sext %{{.*}} : vector<3xi32> to vector<3xi64>
  %0 = spv.SConvert %arg0: vector<3xi32> to vector<3xi64>

  // CHECK: llvm.trunc %{{.*}} : vector<3xi64> to vector<3xi32>
  %1 = spv.SConvert %arg1: vector<3xi64> to vector<3xi32>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.UConvert
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @uconvert_scalar
spv.func @uconvert_scalar(%arg0: i32, %arg1: i64) "None" {
  // CHECK: llvm.zext %{{.*}} : i32 to i64
  %0 = spv.UConvert %arg0: i32 to i64

  // CHECK: llvm.trunc %{{.*}} : i64 to i32
  %1 = spv.UConvert %arg1: i64 to i32
  spv.Return
}

// CHECK-LABEL: @uconvert_vector
spv.func @uconvert_vector(%arg0: vector<3xi32>, %arg1: vector<3xi64>) "None" {
  // CHECK: llvm.zext %{{.*}} : vector<3xi32> to vector<3xi64>
  %0 = spv.UConvert %arg0: vector<3xi32> to vector<3xi64>

  // CHECK: llvm.trunc %{{.*}} : vector<3xi64> to vector<3xi32>
  %1 = spv.UConvert %arg1: vector<3xi64> to vector<3xi32>
  spv.Return
}

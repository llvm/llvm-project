// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.Bitcast
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @bitcast_float_to_integer_scalar
spirv.func @bitcast_float_to_integer_scalar(%arg0 : f32) "None" {
  // CHECK: llvm.bitcast {{.*}} : f32 to i32
  %0 = spirv.Bitcast %arg0: f32 to i32
  spirv.Return
}

// CHECK-LABEL: @bitcast_float_to_integer_vector
spirv.func @bitcast_float_to_integer_vector(%arg0 : vector<3xf32>) "None" {
  // CHECK: {{.*}} = llvm.bitcast {{.*}} : vector<3xf32> to vector<3xi32>
  %0 = spirv.Bitcast %arg0: vector<3xf32> to vector<3xi32>
  spirv.Return
}

// CHECK-LABEL: @bitcast_vector_to_scalar
spirv.func @bitcast_vector_to_scalar(%arg0 : vector<2xf32>) "None" {
  // CHECK: {{.*}} = llvm.bitcast {{.*}} : vector<2xf32> to i64
  %0 = spirv.Bitcast %arg0: vector<2xf32> to i64
  spirv.Return
}

// CHECK-LABEL: @bitcast_scalar_to_vector
spirv.func @bitcast_scalar_to_vector(%arg0 : f64) "None" {
  // CHECK: {{.*}} = llvm.bitcast {{.*}} : f64 to vector<2xi32>
  %0 = spirv.Bitcast %arg0: f64 to vector<2xi32>
  spirv.Return
}

// CHECK-LABEL: @bitcast_vector_to_vector
spirv.func @bitcast_vector_to_vector(%arg0 : vector<4xf32>) "None" {
  // CHECK: {{.*}} = llvm.bitcast {{.*}} : vector<4xf32> to vector<2xi64>
  %0 = spirv.Bitcast %arg0: vector<4xf32> to vector<2xi64>
  spirv.Return
}

// CHECK-LABEL: @bitcast_pointer
spirv.func @bitcast_pointer(%arg0: !spirv.ptr<f32, Function>) "None" {
  // CHECK-NOT: llvm.bitcast
  %0 = spirv.Bitcast %arg0 : !spirv.ptr<f32, Function> to !spirv.ptr<i32, Function>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.ConvertFToS
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @convert_float_to_signed_scalar
spirv.func @convert_float_to_signed_scalar(%arg0: f32) "None" {
  // CHECK: llvm.fptosi %{{.*}} : f32 to i32
  %0 = spirv.ConvertFToS %arg0: f32 to i32
  spirv.Return
}

// CHECK-LABEL: @convert_float_to_signed_vector
spirv.func @convert_float_to_signed_vector(%arg0: vector<2xf32>) "None" {
  // CHECK: llvm.fptosi %{{.*}} : vector<2xf32> to vector<2xi32>
    %0 = spirv.ConvertFToS %arg0: vector<2xf32> to vector<2xi32>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.ConvertFToU
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @convert_float_to_unsigned_scalar
spirv.func @convert_float_to_unsigned_scalar(%arg0: f32) "None" {
  // CHECK: llvm.fptoui %{{.*}} : f32 to i32
  %0 = spirv.ConvertFToU %arg0: f32 to i32
  spirv.Return
}

// CHECK-LABEL: @convert_float_to_unsigned_vector
spirv.func @convert_float_to_unsigned_vector(%arg0: vector<2xf32>) "None" {
  // CHECK: llvm.fptoui %{{.*}} : vector<2xf32> to vector<2xi32>
    %0 = spirv.ConvertFToU %arg0: vector<2xf32> to vector<2xi32>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.ConvertSToF
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @convert_signed_to_float_scalar
spirv.func @convert_signed_to_float_scalar(%arg0: i32) "None" {
  // CHECK: llvm.sitofp %{{.*}} : i32 to f32
  %0 = spirv.ConvertSToF %arg0: i32 to f32
  spirv.Return
}

// CHECK-LABEL: @convert_signed_to_float_vector
spirv.func @convert_signed_to_float_vector(%arg0: vector<3xi32>) "None" {
  // CHECK: llvm.sitofp %{{.*}} : vector<3xi32> to vector<3xf32>
    %0 = spirv.ConvertSToF %arg0: vector<3xi32> to vector<3xf32>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.ConvertUToF
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @convert_unsigned_to_float_scalar
spirv.func @convert_unsigned_to_float_scalar(%arg0: i32) "None" {
  // CHECK: llvm.uitofp %{{.*}} : i32 to f32
  %0 = spirv.ConvertUToF %arg0: i32 to f32
  spirv.Return
}

// CHECK-LABEL: @convert_unsigned_to_float_vector
spirv.func @convert_unsigned_to_float_vector(%arg0: vector<3xi32>) "None" {
  // CHECK: llvm.uitofp %{{.*}} : vector<3xi32> to vector<3xf32>
    %0 = spirv.ConvertUToF %arg0: vector<3xi32> to vector<3xf32>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.FConvert
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fconvert_scalar
spirv.func @fconvert_scalar(%arg0: f32, %arg1: f64) "None" {
  // CHECK: llvm.fpext %{{.*}} : f32 to f64
  %0 = spirv.FConvert %arg0: f32 to f64

  // CHECK: llvm.fptrunc %{{.*}} : f64 to f32
  %1 = spirv.FConvert %arg1: f64 to f32
  spirv.Return
}

// CHECK-LABEL: @fconvert_vector
spirv.func @fconvert_vector(%arg0: vector<2xf32>, %arg1: vector<2xf64>) "None" {
  // CHECK: llvm.fpext %{{.*}} : vector<2xf32> to vector<2xf64>
  %0 = spirv.FConvert %arg0: vector<2xf32> to vector<2xf64>

  // CHECK: llvm.fptrunc %{{.*}} : vector<2xf64> to vector<2xf32>
  %1 = spirv.FConvert %arg1: vector<2xf64> to vector<2xf32>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.SConvert
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @sconvert_scalar
spirv.func @sconvert_scalar(%arg0: i32, %arg1: i64) "None" {
  // CHECK: llvm.sext %{{.*}} : i32 to i64
  %0 = spirv.SConvert %arg0: i32 to i64

  // CHECK: llvm.trunc %{{.*}} : i64 to i32
  %1 = spirv.SConvert %arg1: i64 to i32
  spirv.Return
}

// CHECK-LABEL: @sconvert_vector
spirv.func @sconvert_vector(%arg0: vector<3xi32>, %arg1: vector<3xi64>) "None" {
  // CHECK: llvm.sext %{{.*}} : vector<3xi32> to vector<3xi64>
  %0 = spirv.SConvert %arg0: vector<3xi32> to vector<3xi64>

  // CHECK: llvm.trunc %{{.*}} : vector<3xi64> to vector<3xi32>
  %1 = spirv.SConvert %arg1: vector<3xi64> to vector<3xi32>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.UConvert
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @uconvert_scalar
spirv.func @uconvert_scalar(%arg0: i32, %arg1: i64) "None" {
  // CHECK: llvm.zext %{{.*}} : i32 to i64
  %0 = spirv.UConvert %arg0: i32 to i64

  // CHECK: llvm.trunc %{{.*}} : i64 to i32
  %1 = spirv.UConvert %arg1: i64 to i32
  spirv.Return
}

// CHECK-LABEL: @uconvert_vector
spirv.func @uconvert_vector(%arg0: vector<3xi32>, %arg1: vector<3xi64>) "None" {
  // CHECK: llvm.zext %{{.*}} : vector<3xi32> to vector<3xi64>
  %0 = spirv.UConvert %arg0: vector<3xi32> to vector<3xi64>

  // CHECK: llvm.trunc %{{.*}} : vector<3xi64> to vector<3xi32>
  %1 = spirv.UConvert %arg1: vector<3xi64> to vector<3xi32>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.ConvertPtrToU
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @convert_ptr_to_u
spirv.func @convert_ptr_to_u(%arg0 : !spirv.ptr<i32, Generic>) "None" {
  // CHECK: llvm.ptrtoint %{{.*}} : !llvm.ptr to i32
  %0 = spirv.ConvertPtrToU %arg0 : !spirv.ptr<i32, Generic> to i32
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.ConvertUToPtr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @convert_u_to_ptr
spirv.func @convert_u_to_ptr(%arg0 : i32) "None" {
  // CHECK: llvm.inttoptr %{{.*}} : i32 to !llvm.ptr
  %0 = spirv.ConvertUToPtr %arg0 : i32 to !spirv.ptr<i32, Generic>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.PtrCastToGeneric
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @ptr_cast_to_generic
spirv.func @ptr_cast_to_generic(%arg0 : !spirv.ptr<f32, CrossWorkgroup>) "None" {
  // CHECK: llvm.addrspacecast %{{.*}} : !llvm.ptr to !llvm.ptr
  %0 = spirv.PtrCastToGeneric %arg0 : !spirv.ptr<f32, CrossWorkgroup> to !spirv.ptr<f32, Generic>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.GenericCastToPtr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @generic_cast_to_ptr
spirv.func @generic_cast_to_ptr(%arg0 : !spirv.ptr<vector<2xi32>, Generic>) "None" {
  // CHECK: llvm.addrspacecast %{{.*}} : !llvm.ptr to !llvm.ptr
  %0 = spirv.GenericCastToPtr %arg0 : !spirv.ptr<vector<2xi32>, Generic> to !spirv.ptr<vector<2xi32>, CrossWorkgroup>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.GenericCastToPtrExplicit
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @generic_cast_to_ptr_explicit
spirv.func @generic_cast_to_ptr_explicit(%arg0 : !spirv.ptr<vector<2xi32>, Generic>) "None" {
  // CHECK: llvm.addrspacecast %{{.*}} : !llvm.ptr to !llvm.ptr
  %0 = spirv.GenericCastToPtrExplicit %arg0 : !spirv.ptr<vector<2xi32>, Generic> to !spirv.ptr<vector<2xi32>, CrossWorkgroup>
  spirv.Return
}

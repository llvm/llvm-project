// RUN: mlir-translate -test-spirv-roundtrip -split-input-file %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  spv.func @bit_cast(%arg0 : f32) "None" {
    // CHECK: {{%.*}} = spv.Bitcast {{%.*}} : f32 to i32
    %0 = spv.Bitcast %arg0 : f32 to i32
    // CHECK: {{%.*}} = spv.Bitcast {{%.*}} : i32 to si32
    %1 = spv.Bitcast %0 : i32 to si32
    // CHECK: {{%.*}} = spv.Bitcast {{%.*}} : si32 to i32
    %2 = spv.Bitcast %1 : si32 to ui32
    spv.Return
  }
}

// -----

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  spv.func @convert_f_to_s(%arg0 : f32) -> i32 "None" {
    // CHECK: {{%.*}} = spv.ConvertFToS {{%.*}} : f32 to i32
    %0 = spv.ConvertFToS %arg0 : f32 to i32
    spv.ReturnValue %0 : i32
  }
  spv.func @convert_f64_to_s32(%arg0 : f64) -> i32 "None" {
    // CHECK: {{%.*}} = spv.ConvertFToS {{%.*}} : f64 to i32
    %0 = spv.ConvertFToS %arg0 : f64 to i32
    spv.ReturnValue %0 : i32
  }
  spv.func @convert_f_to_u(%arg0 : f32) -> i32 "None" {
    // CHECK: {{%.*}} = spv.ConvertFToU {{%.*}} : f32 to i32
    %0 = spv.ConvertFToU %arg0 : f32 to i32
    spv.ReturnValue %0 : i32
  }
  spv.func @convert_f64_to_u32(%arg0 : f64) -> i32 "None" {
    // CHECK: {{%.*}} = spv.ConvertFToU {{%.*}} : f64 to i32
    %0 = spv.ConvertFToU %arg0 : f64 to i32
    spv.ReturnValue %0 : i32
  }
  spv.func @convert_s_to_f(%arg0 : i32) -> f32 "None" {
    // CHECK: {{%.*}} = spv.ConvertSToF {{%.*}} : i32 to f32
    %0 = spv.ConvertSToF %arg0 : i32 to f32
    spv.ReturnValue %0 : f32
  }
  spv.func @convert_s64_to_f32(%arg0 : i64) -> f32 "None" {
    // CHECK: {{%.*}} = spv.ConvertSToF {{%.*}} : i64 to f32
    %0 = spv.ConvertSToF %arg0 : i64 to f32
    spv.ReturnValue %0 : f32
  }
  spv.func @convert_u_to_f(%arg0 : i32) -> f32 "None" {
    // CHECK: {{%.*}} = spv.ConvertUToF {{%.*}} : i32 to f32
    %0 = spv.ConvertUToF %arg0 : i32 to f32
    spv.ReturnValue %0 : f32
  }
  spv.func @convert_u64_to_f32(%arg0 : i64) -> f32 "None" {
    // CHECK: {{%.*}} = spv.ConvertUToF {{%.*}} : i64 to f32
    %0 = spv.ConvertUToF %arg0 : i64 to f32
    spv.ReturnValue %0 : f32
  }
  spv.func @f_convert(%arg0 : f32) -> f64 "None" {
    // CHECK: {{%.*}} = spv.FConvert {{%.*}} : f32 to f64
    %0 = spv.FConvert %arg0 : f32 to f64
    spv.ReturnValue %0 : f64
  }
  spv.func @s_convert(%arg0 : i32) -> i64 "None" {
    // CHECK: {{%.*}} = spv.SConvert {{%.*}} : i32 to i64
    %0 = spv.SConvert %arg0 : i32 to i64
    spv.ReturnValue %0 : i64
  }
  spv.func @u_convert(%arg0 : i32) -> i64 "None" {
    // CHECK: {{%.*}} = spv.UConvert {{%.*}} : i32 to i64
    %0 = spv.UConvert %arg0 : i32 to i64
    spv.ReturnValue %0 : i64
  }
}

// -----

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Kernel], []> {
  spv.func @ptr_cast_to_generic(%arg0 : !spv.ptr<f32, CrossWorkgroup>) "None" {
    // CHECK: {{%.*}} = spv.PtrCastToGeneric {{%.*}} : !spv.ptr<f32, CrossWorkgroup> to !spv.ptr<f32, Generic>
    %0 = spv.PtrCastToGeneric %arg0 : !spv.ptr<f32, CrossWorkgroup> to !spv.ptr<f32, Generic>
    spv.Return
  }
  spv.func @generic_cast_to_ptr(%arg0 : !spv.ptr<vector<2xi32>, Generic>) "None" {
    // CHECK: {{%.*}} = spv.GenericCastToPtr {{%.*}} : !spv.ptr<vector<2xi32>, Generic> to !spv.ptr<vector<2xi32>, CrossWorkgroup>
    %0 = spv.GenericCastToPtr %arg0 : !spv.ptr<vector<2xi32>, Generic> to !spv.ptr<vector<2xi32>, CrossWorkgroup>
    spv.Return
  }
  spv.func @generic_cast_to_ptr_explicit(%arg0 : !spv.ptr<vector<2xi32>, Generic>) "None" {
    // CHECK: {{%.*}} = spv.GenericCastToPtrExplicit {{%.*}} : !spv.ptr<vector<2xi32>, Generic> to !spv.ptr<vector<2xi32>, CrossWorkgroup>
    %0 = spv.GenericCastToPtrExplicit %arg0 : !spv.ptr<vector<2xi32>, Generic> to !spv.ptr<vector<2xi32>, CrossWorkgroup>
    spv.Return
  }
}

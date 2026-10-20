// RUN: mlir-translate --no-implicit-module --split-input-file --test-spirv-roundtrip %s | FileCheck %s

// RUN: %if spirv-tools %{ rm -rf %t %}
// RUN: %if spirv-tools %{ mkdir %t %}
// RUN: %if spirv-tools %{ mlir-translate --no-implicit-module --serialize-spirv --split-input-file --spirv-save-validation-files-with-prefix=%t/module %s %}
// RUN: %if spirv-tools %{ spirv-val %t %}

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], []> {
  // CHECK: location = 0 : i32
  spirv.GlobalVariable @var {location = 0 : i32} : !spirv.ptr<vector<4xf32>, Input>
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], []> {
  // CHECK: no_perspective
  spirv.GlobalVariable @var {no_perspective} : !spirv.ptr<vector<4xf32>, Input>
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], []> {
  // CHECK: flat
  spirv.GlobalVariable @var {flat} : !spirv.ptr<si32, Input>
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], [SPV_KHR_variable_pointers]> {
  // CHECK: aliased
  // CHECK: aliased
  spirv.GlobalVariable @var1 bind(0, 0) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.array<4xf32, stride=4>[0])>, StorageBuffer>
  spirv.GlobalVariable @var2 bind(0, 0) {aliased} : !spirv.ptr<!spirv.struct<(vector<4xf32>[0])>, StorageBuffer>
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], [SPV_KHR_variable_pointers]> {
  // CHECK: non_readable
  spirv.GlobalVariable @var bind(0, 0) {non_readable} : !spirv.ptr<!spirv.struct<(!spirv.array<4xf32, stride=4>[0])>, StorageBuffer>
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], [SPV_KHR_variable_pointers]> {
  // CHECK: non_writable
  spirv.GlobalVariable @var bind(0, 0) {non_writable} : !spirv.ptr<!spirv.struct<(!spirv.array<4xf32, stride=4>[0])>, StorageBuffer>
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], [SPV_KHR_variable_pointers]> {
  // CHECK: restrict
  spirv.GlobalVariable @var bind(0, 0) {restrict} : !spirv.ptr<!spirv.struct<(!spirv.array<4xf32, stride=4>[0])>, StorageBuffer>
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], []> {
  // CHECK: relaxed_precision
  spirv.GlobalVariable @var {location = 0 : i32, relaxed_precision} : !spirv.ptr<vector<4xf32>, Output>
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Tessellation, Linkage], []> {
  // CHECK: patch
  spirv.GlobalVariable @var {patch} : !spirv.ptr<vector<4xf32>, Input>
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], []> {
  // CHECK: invariant
  spirv.GlobalVariable @var {invariant} : !spirv.ptr<vector<2xf32>, Output>
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], []> {
  // CHECK: coherent
  spirv.GlobalVariable @var {coherent} : !spirv.ptr<vector<2xf32>, Output>
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], []> {
  // CHECK: volatile
  spirv.GlobalVariable @var {volatile} : !spirv.ptr<vector<2xf32>, Output>
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], []> {
  // CHECK: index = 42
  spirv.GlobalVariable @var {index = 42 : i32} : !spirv.ptr<vector<2xf32>, Output>
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage, TransformFeedback], []> {
  // CHECK: offset = 0
  // CHECK-SAME: xfb_buffer = 0
  // CHECK-SAME: xfb_stride = 112
  spirv.GlobalVariable @spirv_var_64 {offset = 0 : i32, xfb_buffer = 0 : i32, xfb_stride = 112 : i32} : !spirv.ptr<vector<3xf32>, Output>
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], []> {
  // CHECK: linkage_attributes = #spirv.linkage_attributes<linkage_name = "outSideGlobalVar1", linkage_type = <Import>>
  spirv.GlobalVariable @var1 {
    linkage_attributes=#spirv.linkage_attributes<
      linkage_name="outSideGlobalVar1", 
      linkage_type=<Import>
    >
  } : !spirv.ptr<f32, Private>
}

// -----

spirv.module Logical OpenCL requires #spirv.vce<v1.0, [Kernel, Linkage], [SPV_KHR_no_integer_wrap_decoration]> {
spirv.func @iadd_decorations(%arg: i32) -> i32 "None" {
  // CHECK: spirv.IAdd %{{.*}}, %{{.*}} {no_signed_wrap, no_unsigned_wrap}
  %0 = spirv.IAdd %arg, %arg {no_signed_wrap, no_unsigned_wrap} : i32
  spirv.ReturnValue %0 : i32
}
}

// -----

spirv.module Logical OpenCL requires #spirv.vce<v1.0, [Kernel, Linkage], []> {
spirv.func @fadd_decorations(%arg: f32) -> f32 "None" {
  // CHECK: spirv.FAdd %{{.*}}, %{{.*}} {fp_fast_math_mode = #spirv.fastmath_mode<NotNaN|NotInf|NSZ>}
  %0 = spirv.FAdd %arg, %arg {fp_fast_math_mode = #spirv.fastmath_mode<NotNaN|NotInf|NSZ>} : f32
  spirv.ReturnValue %0 : f32
}
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], []> {
spirv.func @fmul_decorations(%arg: f32) -> f32 "None" {
  // CHECK: spirv.FMul %{{.*}}, %{{.*}} {no_contraction}
  %0 = spirv.FMul %arg, %arg {no_contraction} : f32
  spirv.ReturnValue %0 : f32
}
}

// -----

spirv.module Logical OpenCL requires #spirv.vce<v1.0, [Kernel, Linkage, Float16], []> {
spirv.func @fp_rounding_mode(%arg: f32) -> f16 "None" {
  // CHECK: spirv.FConvert %arg0 {fp_rounding_mode = #spirv.fp_rounding_mode<RTN>} : f32 to f16
  %0 = spirv.FConvert %arg {fp_rounding_mode = #spirv.fp_rounding_mode<RTN>} : f32 to f16
  spirv.ReturnValue %0 : f16
}
}

// -----

spirv.module Logical OpenCL requires #spirv.vce<v1.0, [Kernel, Linkage, Float64], []> {
spirv.func @f_convert_rounding_mode(%arg: f64) -> f32 "None" {
  // CHECK: spirv.FConvert %{{.*}} {fp_rounding_mode = #spirv.fp_rounding_mode<RTZ>} : f64 to f32
  %0 = spirv.FConvert %arg {fp_rounding_mode = #spirv.fp_rounding_mode<RTZ>} : f64 to f32
  spirv.ReturnValue %0 : f32
}
spirv.func @convert_s_to_f_rounding_mode(%arg: i32) -> f32 "None" {
  // CHECK: spirv.ConvertSToF %{{.*}} {fp_rounding_mode = #spirv.fp_rounding_mode<RTE>} : i32 to f32
  %0 = spirv.ConvertSToF %arg {fp_rounding_mode = #spirv.fp_rounding_mode<RTE>} : i32 to f32
  spirv.ReturnValue %0 : f32
}
spirv.func @convert_u_to_f_rounding_mode(%arg: vector<4xi32>) -> vector<4xf32> "None" {
  // CHECK: spirv.ConvertUToF %{{.*}} {fp_rounding_mode = #spirv.fp_rounding_mode<RTP>} : vector<4xi32> to vector<4xf32>
  %0 = spirv.ConvertUToF %arg {fp_rounding_mode = #spirv.fp_rounding_mode<RTP>} : vector<4xi32> to vector<4xf32>
  spirv.ReturnValue %0 : vector<4xf32>
}
spirv.func @convert_f_to_s_rounding_mode(%arg: f32) -> i32 "None" {
  // CHECK: spirv.ConvertFToS %{{.*}} {fp_rounding_mode = #spirv.fp_rounding_mode<RTN>} : f32 to i32
  %0 = spirv.ConvertFToS %arg {fp_rounding_mode = #spirv.fp_rounding_mode<RTN>} : f32 to i32
  spirv.ReturnValue %0 : i32
}
spirv.func @convert_f_to_u_rounding_mode(%arg: f32) -> i32 "None" {
  // CHECK: spirv.ConvertFToU %{{.*}} {fp_rounding_mode = #spirv.fp_rounding_mode<RTE>} : f32 to i32
  %0 = spirv.ConvertFToU %arg {fp_rounding_mode = #spirv.fp_rounding_mode<RTE>} : f32 to i32
  spirv.ReturnValue %0 : i32
}
}

// -----

spirv.module Logical OpenCL requires #spirv.vce<v1.0, [Kernel, Linkage, Int8, Int64], []> {
spirv.func @convert_f_to_s_saturated(%arg: f32) -> i32 "None" {
  // CHECK: spirv.ConvertFToS %{{.*}} {saturated_conversion} : f32 to i32
  %0 = spirv.ConvertFToS %arg {saturated_conversion} : f32 to i32
  spirv.ReturnValue %0 : i32
}
spirv.func @convert_f_to_u_saturated(%arg: vector<4xf32>) -> vector<4xi32> "None" {
  // CHECK: spirv.ConvertFToU %{{.*}} {saturated_conversion} : vector<4xf32> to vector<4xi32>
  %0 = spirv.ConvertFToU %arg {saturated_conversion} : vector<4xf32> to vector<4xi32>
  spirv.ReturnValue %0 : vector<4xi32>
}
spirv.func @s_convert_saturated(%arg: i64) -> i32 "None" {
  // CHECK: spirv.SConvert %{{.*}} {saturated_conversion} : i64 to i32
  %0 = spirv.SConvert %arg {saturated_conversion} : i64 to i32
  spirv.ReturnValue %0 : i32
}
spirv.func @u_convert_saturated(%arg: i32) -> i8 "None" {
  // CHECK: spirv.UConvert %{{.*}} {saturated_conversion} : i32 to i8
  %0 = spirv.UConvert %arg {saturated_conversion} : i32 to i8
  spirv.ReturnValue %0 : i8
}
spirv.func @convert_f_to_s_saturated_rounding_mode(%arg: f32) -> i32 "None" {
  // CHECK: spirv.ConvertFToS %{{.*}} {fp_rounding_mode = #spirv.fp_rounding_mode<RTE>, saturated_conversion} : f32 to i32
  %0 = spirv.ConvertFToS %arg {fp_rounding_mode = #spirv.fp_rounding_mode<RTE>, saturated_conversion} : f32 to i32
  spirv.ReturnValue %0 : i32
}
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], []> {
  // CHECK: spirv.func @relaxed_precision_arg({{%.*}}: !spirv.ptr<f32, Function> {spirv.decoration = #spirv.decoration<RelaxedPrecision>}) "None" attributes {relaxed_precision} {
  spirv.func @relaxed_precision_arg(%arg0: !spirv.ptr<f32, Function> {spirv.decoration = #spirv.decoration<RelaxedPrecision>}) -> () "None" attributes {relaxed_precision} {
    spirv.Return
  }
}

// -----

spirv.module Physical64 OpenCL requires #spirv.vce<v1.0, [Kernel, Addresses, Int8, Int64, Linkage], []> {
  // CHECK: spirv.func @cpacked_struct(%{{.*}}: !spirv.ptr<!spirv.struct<(i8, i64, i32), CPacked>, CrossWorkgroup>)
  spirv.func @cpacked_struct(%arg0: !spirv.ptr<!spirv.struct<(i8, i64, i32), CPacked>, CrossWorkgroup>) "None" {
    spirv.Return
  }

  // CHECK: spirv.func @cpacked_identified_struct(%{{.*}}: !spirv.ptr<!spirv.struct<packed, (i8, i64, i32), CPacked>, CrossWorkgroup>)
  spirv.func @cpacked_identified_struct(%arg0: !spirv.ptr<!spirv.struct<packed, (i8, i64, i32), CPacked>, CrossWorkgroup>) "None" {
    spirv.Return
  }
}

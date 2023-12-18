// RUN: mlir-translate -no-implicit-module -split-input-file -test-spirv-roundtrip %s | FileCheck %s

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  // CHECK: location = 0 : i32
  spirv.GlobalVariable @var {location = 0 : i32} : !spirv.ptr<vector<4xf32>, Input>
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  // CHECK: no_perspective
  spirv.GlobalVariable @var {no_perspective} : !spirv.ptr<vector<4xf32>, Input>
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  // CHECK: flat
  spirv.GlobalVariable @var {flat} : !spirv.ptr<si32, Input>
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  // CHECK: aliased
  // CHECK: aliased
  spirv.GlobalVariable @var1 bind(0, 0) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.array<4xf32, stride=4>[0])>, StorageBuffer>
  spirv.GlobalVariable @var2 bind(0, 0) {aliased} : !spirv.ptr<!spirv.struct<(vector<4xf32>[0])>, StorageBuffer>
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  // CHECK: non_readable
  spirv.GlobalVariable @var bind(0, 0) {non_readable} : !spirv.ptr<!spirv.struct<(!spirv.array<4xf32, stride=4>[0])>, StorageBuffer>
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  // CHECK: non_writable
  spirv.GlobalVariable @var bind(0, 0) {non_writable} : !spirv.ptr<!spirv.struct<(!spirv.array<4xf32, stride=4>[0])>, StorageBuffer>
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  // CHECK: restrict
  spirv.GlobalVariable @var bind(0, 0) {restrict} : !spirv.ptr<!spirv.struct<(!spirv.array<4xf32, stride=4>[0])>, StorageBuffer>
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  // CHECK: relaxed_precision
  spirv.GlobalVariable @var {location = 0 : i32, relaxed_precision} : !spirv.ptr<vector<4xf32>, Output>
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], []> {
  // CHECK: linkage_attributes = #spirv.linkage_attributes<linkage_name = outSideGlobalVar1, linkage_type = <Import>>
  spirv.GlobalVariable @var1 {
    linkage_attributes=#spirv.linkage_attributes<
      linkage_name="outSideGlobalVar1", 
      linkage_type=<Import>
    >
  } : !spirv.ptr<f32, Private>
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Kernel], []> {
spirv.func @iadd_decorations(%arg: i32) -> i32 "None" {
  // CHECK: spirv.IAdd %{{.*}}, %{{.*}} {no_signed_wrap, no_unsigned_wrap}
  %0 = spirv.IAdd %arg, %arg {no_signed_wrap, no_unsigned_wrap} : i32
  spirv.ReturnValue %0 : i32
}
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Kernel], []> {
spirv.func @fadd_decorations(%arg: f32) -> f32 "None" {
  // CHECK: spirv.FAdd %{{.*}}, %{{.*}} {fp_fast_math_mode = #spirv.fastmath_mode<NotNaN|NotInf|NSZ>}
  %0 = spirv.FAdd %arg, %arg {fp_fast_math_mode = #spirv.fastmath_mode<NotNaN|NotInf|NSZ>} : f32
  spirv.ReturnValue %0 : f32
}
}

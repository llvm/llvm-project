// RUN: mlir-translate -no-implicit-module -split-input-file -test-spirv-roundtrip -verify-diagnostics %s | FileCheck %s

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
  // CHECK: linkage_attributes = #spirv.linkage_attributes<linkage_name = "outSideGlobalVar1", linkage_type = <Import>>
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

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Kernel], []> {
spirv.func @fmul_decorations(%arg: f32) -> f32 "None" {
  // CHECK: spirv.FMul %{{.*}}, %{{.*}} {no_contraction}
  %0 = spirv.FMul %arg, %arg {no_contraction} : f32
  spirv.ReturnValue %0 : f32
}
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Kernel, Float16], []> {
spirv.func @fp_rounding_mode(%arg: f32) -> f16 "None" {
  // CHECK: spirv.FConvert %arg0 {fp_rounding_mode = #spirv.fp_rounding_mode<RTN>} : f32 to f16
  %0 = spirv.FConvert %arg {fp_rounding_mode = #spirv.fp_rounding_mode<RTN>} : f32 to f16
  spirv.ReturnValue %0 : f16
}
}

// -----

// CHECK-LABEL: spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [CacheControlsINTEL], [SPV_INTEL_cache_controls]> {

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [CacheControlsINTEL], [SPV_INTEL_cache_controls]> {
  spirv.func @cache_controls() "None" {
    // CHECK: spirv.Variable {cache_control_load_intel = [#spirv.cache_control_load_intel<cache_level = 0, load_cache_control = Uncached>, #spirv.cache_control_load_intel<cache_level = 1, load_cache_control = Cached>, #spirv.cache_control_load_intel<cache_level = 2, load_cache_control = InvalidateAfterR>]} : !spirv.ptr<f32, Function>
    %0 = spirv.Variable {cache_control_load_intel = [#spirv.cache_control_load_intel<cache_level = 0, load_cache_control = Uncached>, #spirv.cache_control_load_intel<cache_level = 1, load_cache_control = Cached>, #spirv.cache_control_load_intel<cache_level = 2, load_cache_control = InvalidateAfterR>]} : !spirv.ptr<f32, Function>
    // CHECK: spirv.Variable {cache_control_store_intel = [#spirv.cache_control_store_intel<cache_level = 0, store_cache_control = Uncached>, #spirv.cache_control_store_intel<cache_level = 1, store_cache_control = WriteThrough>, #spirv.cache_control_store_intel<cache_level = 2, store_cache_control = WriteBack>]} : !spirv.ptr<f32, Function>
    %1 = spirv.Variable {cache_control_store_intel = [#spirv.cache_control_store_intel<cache_level = 0, store_cache_control = Uncached>, #spirv.cache_control_store_intel<cache_level = 1, store_cache_control = WriteThrough>, #spirv.cache_control_store_intel<cache_level = 2, store_cache_control = WriteBack>]} : !spirv.ptr<f32, Function>
    spirv.Return
  }
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [CacheControlsINTEL], [SPV_INTEL_cache_controls]> {
  spirv.func @cache_controls_invalid_type() "None" {
    // expected-error@below {{expecting array attribute of CacheControlLoadINTEL for CacheControlLoadINTEL}}
    %0 = spirv.Variable {cache_control_load_intel = #spirv.cache_control_load_intel<cache_level = 0, load_cache_control = Uncached>} : !spirv.ptr<f32, Function>
    spirv.Return
  }
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [CacheControlsINTEL], [SPV_INTEL_cache_controls]> {
  spirv.func @cache_controls_invalid_type() "None" {
    // expected-error@below {{expecting array attribute of CacheControlStoreINTEL for CacheControlStoreINTEL}}
    %0 = spirv.Variable {cache_control_store_intel = [#spirv.cache_control_store_intel<cache_level = 0, store_cache_control = Uncached>, 0 : i32]} : !spirv.ptr<f32, Function>
    spirv.Return
  }
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [CacheControlsINTEL], [SPV_INTEL_cache_controls]> {
  spirv.func @cache_controls_invalid_type() "None" {
    // expected-error@below {{expecting non-empty array attribute of CacheControlStoreINTEL for CacheControlStoreINTEL}}
    %0 = spirv.Variable {cache_control_store_intel = []} : !spirv.ptr<f32, Function>
    spirv.Return
  }
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  // CHECK: spirv.func @relaxed_precision_arg({{%.*}}: !spirv.ptr<f32, Function> {spirv.decoration = #spirv.decoration<RelaxedPrecision>}) "None" attributes {relaxed_precision} {
  spirv.func @relaxed_precision_arg(%arg0: !spirv.ptr<f32, Function> {spirv.decoration = #spirv.decoration<RelaxedPrecision>}) -> () "None" attributes {relaxed_precision} {
    spirv.Return
  }
}

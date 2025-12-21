// RUN: mlir-translate --no-implicit-module --split-input-file --test-spirv-roundtrip --verify-diagnostics %s | FileCheck %s

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

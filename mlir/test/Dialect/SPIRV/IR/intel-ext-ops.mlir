// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.INTEL.ConvertFToBF16
//===----------------------------------------------------------------------===//

spirv.func @f32_to_bf16(%arg0 : f32) "None" {
  // CHECK: {{%.*}} = spirv.INTEL.ConvertFToBF16 {{%.*}} : f32 to i16
  %0 = spirv.INTEL.ConvertFToBF16 %arg0 : f32 to i16
  spirv.Return
}

// -----

spirv.func @f32_to_bf16_vec(%arg0 : vector<2xf32>) "None" {
  // CHECK: {{%.*}} = spirv.INTEL.ConvertFToBF16 {{%.*}} : vector<2xf32> to vector<2xi16>
  %0 = spirv.INTEL.ConvertFToBF16 %arg0 : vector<2xf32> to vector<2xi16>
  spirv.Return
}

// -----

spirv.func @f32_to_bf16_unsupported(%arg0 : f64) "None" {
  // expected-error @+1 {{operand #0 must be Float32 or vector of Float32 values of length 2/3/4/8/16, but got}}
  %0 = spirv.INTEL.ConvertFToBF16 %arg0 : f64 to i16
  spirv.Return
}

// -----

spirv.func @f32_to_bf16_vec_unsupported(%arg0 : vector<2xf32>) "None" {
  // expected-error @+1 {{operand and result must have same number of elements}}
  %0 = spirv.INTEL.ConvertFToBF16 %arg0 : vector<2xf32> to vector<4xi16>
  spirv.Return
}

// -----

//===----------------------------------------------------------------------===//
// spirv.INTEL.ConvertBF16ToF
//===----------------------------------------------------------------------===//

spirv.func @bf16_to_f32(%arg0 : i16) "None" {
  // CHECK: {{%.*}} = spirv.INTEL.ConvertBF16ToF {{%.*}} : i16 to f32
  %0 = spirv.INTEL.ConvertBF16ToF %arg0 : i16 to f32
  spirv.Return
}

// -----

spirv.func @bf16_to_f32_vec(%arg0 : vector<2xi16>) "None" {
    // CHECK: {{%.*}} = spirv.INTEL.ConvertBF16ToF {{%.*}} : vector<2xi16> to vector<2xf32>
    %0 = spirv.INTEL.ConvertBF16ToF %arg0 : vector<2xi16> to vector<2xf32>
    spirv.Return
}

// -----

spirv.func @bf16_to_f32_unsupported(%arg0 : i16) "None" {
  // expected-error @+1 {{result #0 must be Float32 or vector of Float32 values of length 2/3/4/8/16, but got}}
  %0 = spirv.INTEL.ConvertBF16ToF %arg0 : i16 to f16
  spirv.Return
}

// -----

spirv.func @bf16_to_f32_vec_unsupported(%arg0 : vector<2xi16>) "None" {
  // expected-error @+1 {{operand and result must have same number of elements}}
  %0 = spirv.INTEL.ConvertBF16ToF %arg0 : vector<2xi16> to vector<3xf32>
  spirv.Return
}

// -----

//===----------------------------------------------------------------------===//
// spirv.INTEL.SplitBarrier
//===----------------------------------------------------------------------===//

spirv.func @split_barrier() "None" {
  // CHECK: spirv.INTEL.ControlBarrierArrive <Workgroup> <Device> <Acquire|UniformMemory>
  spirv.INTEL.ControlBarrierArrive <Workgroup> <Device> <Acquire|UniformMemory>
  // CHECK: spirv.INTEL.ControlBarrierWait <Workgroup> <Device> <Acquire|UniformMemory>
  spirv.INTEL.ControlBarrierWait <Workgroup> <Device> <Acquire|UniformMemory>
  spirv.Return
}

// -----

//===----------------------------------------------------------------------===//
// spirv.INTEL.CacheControls
//===----------------------------------------------------------------------===//

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [CacheControlsINTEL], [SPV_INTEL_cache_controls]> {
  spirv.func @foo() "None" {
    // CHECK: spirv.Variable {cache_control_load_intel = [#spirv.cache_control_load_intel<cache_level = 0, load_cache_control = Uncached>, #spirv.cache_control_load_intel<cache_level = 1, load_cache_control = Cached>, #spirv.cache_control_load_intel<cache_level = 2, load_cache_control = InvalidateAfterR>]} : !spirv.ptr<f32, Function>
    %0 = spirv.Variable {cache_control_load_intel = [#spirv.cache_control_load_intel<cache_level = 0, load_cache_control = Uncached>, #spirv.cache_control_load_intel<cache_level = 1, load_cache_control = Cached>, #spirv.cache_control_load_intel<cache_level = 2, load_cache_control = InvalidateAfterR>]} : !spirv.ptr<f32, Function>
    // CHECK: spirv.Variable {cache_control_store_intel = [#spirv.cache_control_store_intel<cache_level = 0, store_cache_control = Uncached>, #spirv.cache_control_store_intel<cache_level = 1, store_cache_control = WriteThrough>, #spirv.cache_control_store_intel<cache_level = 2, store_cache_control = WriteBack>]} : !spirv.ptr<f32, Function>
    %1 = spirv.Variable {cache_control_store_intel = [#spirv.cache_control_store_intel<cache_level = 0, store_cache_control = Uncached>, #spirv.cache_control_store_intel<cache_level = 1, store_cache_control = WriteThrough>, #spirv.cache_control_store_intel<cache_level = 2, store_cache_control = WriteBack>]} : !spirv.ptr<f32, Function>
    spirv.Return
  }
}

// -----

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
  // expected-error @+1 {{operand #0 must be Float32 or fixed-length vector of Float32 values of length 2/3/4/8/16 of ranks 1, but got}}
  %0 = spirv.INTEL.ConvertFToBF16 %arg0 : f64 to i16
  spirv.Return
}

// -----

spirv.func @f32_to_bf16_vec_unsupported(%arg0 : vector<2xf32>) "None" {
  // expected-error @+1 {{op requires the same shape for all operands and results}}
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
  // expected-error @+1 {{result #0 must be Float32 or fixed-length vector of Float32 values of length 2/3/4/8/16 of ranks 1, but got}}
  %0 = spirv.INTEL.ConvertBF16ToF %arg0 : i16 to f16
  spirv.Return
}

// -----

spirv.func @bf16_to_f32_vec_unsupported(%arg0 : vector<2xi16>) "None" {
  // expected-error @+1 {{op requires the same shape for all operands and results}}
  %0 = spirv.INTEL.ConvertBF16ToF %arg0 : vector<2xi16> to vector<3xf32>
  spirv.Return
}

// -----

//===----------------------------------------------------------------------===//
// spirv.INTEL.RoundFToTF32
//===----------------------------------------------------------------------===//

spirv.func @f32_to_tf32(%arg0 : f32) "None" {
  // CHECK: {{%.*}} = spirv.INTEL.RoundFToTF32 {{%.*}} : f32 to f32
  %0 = spirv.INTEL.RoundFToTF32 %arg0 : f32 to f32
  spirv.Return
}

// -----

spirv.func @f32_to_tf32_vec(%arg0 : vector<2xf32>) "None" {
  // CHECK: {{%.*}} = spirv.INTEL.RoundFToTF32 {{%.*}} : vector<2xf32> to vector<2xf32>
  %0 = spirv.INTEL.RoundFToTF32 %arg0 : vector<2xf32> to vector<2xf32>
  spirv.Return
}

// -----

spirv.func @f32_to_tf32_unsupported(%arg0 : f64) "None" {
  // expected-error @+1 {{op operand #0 must be Float32 or fixed-length vector of Float32 values of length 2/3/4/8/16 of ranks 1, but got 'f64'}}
  %0 = spirv.INTEL.RoundFToTF32 %arg0 : f64 to f32
  spirv.Return
}

// -----

spirv.func @f32_to_tf32_vec_unsupported(%arg0 : vector<2xf32>) "None" {
  // expected-error @+1 {{op requires the same shape for all operands and results}}
  %0 = spirv.INTEL.RoundFToTF32 %arg0 : vector<2xf32> to vector<4xf32>
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

//===----------------------------------------------------------------------===//
// spirv.INTEL.MaskedGather
//===----------------------------------------------------------------------===//

spirv.func @masked_gather(
    %ptrs : vector<4x!spirv.ptr<f32, CrossWorkgroup>>,
    %alignment : i32,
    %mask : vector<4xi1>,
    %fill : vector<4xf32>) "None" {
  // CHECK: {{%.*}} = spirv.INTEL.MaskedGather {{%.*}}, {{%.*}}, {{%.*}}, {{%.*}} : vector<4x!spirv.ptr<f32, CrossWorkgroup>>, i32, vector<4xi1>, vector<4xf32> -> vector<4xf32>
  %0 = spirv.INTEL.MaskedGather %ptrs, %alignment, %mask, %fill
       : vector<4x!spirv.ptr<f32, CrossWorkgroup>>, i32,
         vector<4xi1>, vector<4xf32> -> vector<4xf32>
  spirv.Return
}

// -----

spirv.func @masked_gather_i32(
    %ptrs : vector<4x!spirv.ptr<i32, CrossWorkgroup>>,
    %alignment : i32,
    %mask : vector<4xi1>,
    %fill : vector<4xi32>) "None" {
  // CHECK: {{%.*}} = spirv.INTEL.MaskedGather {{%.*}}, {{%.*}}, {{%.*}}, {{%.*}} : vector<4x!spirv.ptr<i32, CrossWorkgroup>>, i32, vector<4xi1>, vector<4xi32> -> vector<4xi32>
  %0 = spirv.INTEL.MaskedGather %ptrs, %alignment, %mask, %fill
       : vector<4x!spirv.ptr<i32, CrossWorkgroup>>, i32,
         vector<4xi1>, vector<4xi32> -> vector<4xi32>
  spirv.Return
}

// -----

spirv.func @masked_gather_pointee_type_mismatch(
    %ptrs : vector<4x!spirv.ptr<f32, CrossWorkgroup>>,
    %alignment : i32,
    %mask : vector<4xi1>,
    %fill : vector<4xi32>) "None" {
  // expected-error @+1 {{'spirv.INTEL.MaskedGather' op failed to verify that pointee type of ptr_vector must match result element type}}
  %0 = spirv.INTEL.MaskedGather %ptrs, %alignment, %mask, %fill
       : vector<4x!spirv.ptr<f32, CrossWorkgroup>>, i32,
         vector<4xi1>, vector<4xi32> -> vector<4xi32>
  spirv.Return
}

// -----

spirv.func @masked_gather_elem_count_mismatch(
    %ptrs : vector<2x!spirv.ptr<f32, CrossWorkgroup>>,
    %alignment : i32,
    %mask : vector<4xi1>,
    %fill : vector<4xf32>) "None" {
  // expected-error @+1 {{'spirv.INTEL.MaskedGather' op failed to verify that pointee type of ptr_vector must match result element type}}
  %0 = spirv.INTEL.MaskedGather %ptrs, %alignment, %mask, %fill
       : vector<2x!spirv.ptr<f32, CrossWorkgroup>>, i32,
         vector<4xi1>, vector<4xf32> -> vector<4xf32>
  spirv.Return
}

// -----

spirv.func @masked_gather_mask_not_bool(
    %ptrs : vector<4x!spirv.ptr<f32, CrossWorkgroup>>,
    %alignment : i32,
    %mask : vector<4xi8>,
    %fill : vector<4xf32>) "None" {
  // expected-error @+1 {{operand #2 must be fixed-length vector of bool values of length 2/3/4/8/16 of ranks 1, but got 'vector<4xi8>'}}
  %0 = spirv.INTEL.MaskedGather %ptrs, %alignment, %mask, %fill
       : vector<4x!spirv.ptr<f32, CrossWorkgroup>>, i32,
         vector<4xi8>, vector<4xf32> -> vector<4xf32>
  spirv.Return
}

// -----

spirv.func @masked_gather_mask_count_mismatch(
    %ptrs : vector<4x!spirv.ptr<f32, CrossWorkgroup>>,
    %alignment : i32,
    %mask : vector<2xi1>,
    %fill : vector<4xf32>) "None" {
  // expected-error @+1 {{'spirv.INTEL.MaskedGather' op failed to verify that mask must be a vector of i1 matching result shape}}
  %0 = spirv.INTEL.MaskedGather %ptrs, %alignment, %mask, %fill
       : vector<4x!spirv.ptr<f32, CrossWorkgroup>>, i32,
         vector<2xi1>, vector<4xf32> -> vector<4xf32>
  spirv.Return
}

// -----

//===----------------------------------------------------------------------===//
// spirv.INTEL.MaskedScatter
//===----------------------------------------------------------------------===//

spirv.func @masked_scatter(
    %ptrs : vector<4x!spirv.ptr<f32, CrossWorkgroup>>,
    %alignment : i32,
    %mask : vector<4xi1>,
    %values : vector<4xf32>) "None" {
  // CHECK: spirv.INTEL.MaskedScatter {{%.*}}, {{%.*}}, {{%.*}}, {{%.*}} : vector<4x!spirv.ptr<f32, CrossWorkgroup>>, i32, vector<4xi1>, vector<4xf32>
  spirv.INTEL.MaskedScatter %ptrs, %alignment, %mask, %values
       : vector<4x!spirv.ptr<f32, CrossWorkgroup>>, i32,
         vector<4xi1>, vector<4xf32>
  spirv.Return
}

// -----

spirv.func @masked_scatter_pointee_mismatch(
    %ptrs : vector<4x!spirv.ptr<i32, CrossWorkgroup>>,
    %alignment : i32,
    %mask : vector<4xi1>,
    %values : vector<4xf32>) "None" {
  // expected-error @+1 {{'spirv.INTEL.MaskedScatter' op failed to verify that pointee type of ptr_vector must match input element type}}
  spirv.INTEL.MaskedScatter %ptrs, %alignment, %mask, %values
       : vector<4x!spirv.ptr<i32, CrossWorkgroup>>, i32,
         vector<4xi1>, vector<4xf32>
  spirv.Return
}

// -----

spirv.func @masked_scatter_mask_count_mismatch(
    %ptrs : vector<4x!spirv.ptr<f32, CrossWorkgroup>>,
    %alignment : i32,
    %mask : vector<2xi1>,
    %values : vector<4xf32>) "None" {
  // expected-error @+1 {{'spirv.INTEL.MaskedScatter' op failed to verify that mask must be a vector of i1 matching input shape}}
  spirv.INTEL.MaskedScatter %ptrs, %alignment, %mask, %values
       : vector<4x!spirv.ptr<f32, CrossWorkgroup>>, i32,
         vector<2xi1>, vector<4xf32>
  spirv.Return
}

// -----

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

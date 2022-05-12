// RUN: mlir-opt -split-input-file -verify-diagnostics %s

func.func @async_cp_memory_space(%dst : memref<16xf32>, %src : memref<16xf32>, %i : index) -> () {
  // expected-error @+1 {{destination memref must have memory space 3}}
  nvgpu.device_async_copy %src[%i], %dst[%i], 16 : memref<16xf32> to memref<16xf32>
  return
}

// -----

func.func @async_cp_memref_type(%dst : memref<16xi32, 3>, %src : memref<16xf32>, %i : index) -> () {
  // expected-error @+1 {{source and destination must have the same element type}}
  nvgpu.device_async_copy %src[%i], %dst[%i], 16 : memref<16xf32> to memref<16xi32, 3>
  return
}

// -----

func.func @async_cp_num_src_indices(%dst : memref<16xf32, 3>, %src : memref<16x16xf32>, %i : index) -> () {
  // expected-error @+1 {{expected 2 source indices, got 1}}
  nvgpu.device_async_copy %src[%i], %dst[%i], 16 : memref<16x16xf32> to memref<16xf32, 3>
  return
}

// -----

func.func @async_cp_num_dst_indices(%dst : memref<16x16xf32, 3>, %src : memref<16xf32>, %i : index) -> () {
  // expected-error @+1 {{expected 2 destination indices, got 1}}
  nvgpu.device_async_copy %src[%i], %dst[%i], 16 : memref<16xf32> to memref<16x16xf32, 3>
  return
}

// -----

func.func @async_cp_num_src_stride(
  %dst : memref<200x100xf32, 3>,
  %src : memref<200x100xf32, affine_map<(d0, d1) -> (200*d0 + 2*d1)>>,
  %i : index) -> () {
  // expected-error @+1 {{source memref most minor dim must have unit stride}}
  nvgpu.device_async_copy %src[%i, %i], %dst[%i, %i], 16 :
    memref<200x100xf32, affine_map<(d0, d1) -> (200*d0 + 2*d1)>> to memref<200x100xf32, 3>
  return
}

// -----

func.func @async_cp_num_dst_stride(
  %dst : memref<200x100xf32, affine_map<(d0, d1) -> (200*d0 + 2*d1)>, 3>,
  %src : memref<200x100xf32>,
  %i : index) -> () {
  // expected-error @+1 {{destination memref most minor dim must have unit stride}}
  nvgpu.device_async_copy %src[%i, %i], %dst[%i, %i], 16 :
    memref<200x100xf32> to memref<200x100xf32, affine_map<(d0, d1) -> (200*d0 + 2*d1)>, 3>
  return
}

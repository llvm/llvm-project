// RUN: mlir-opt -split-input-file -verify-diagnostics %s

func.func @ldmatrix_address_space_f16_x4(%arg0: memref<128x128xf16, 2>) ->  vector<4x1xf16> {
  %c0  = arith.constant 0 : index
  // expected-error @below {{expected nvgpu.ldmatrix srcMemref must have a memory space attribute of IntegerAttr(3) or gpu::AddressSpaceAttr(Workgroup)}}
  %a = nvgpu.ldmatrix %arg0[%c0, %c0] {transpose = false, numTiles = 4 : i32} : memref<128x128xf16, 2> -> vector<4x1xf16>
  return %a : vector<4x1xf16>
}
// -----

func.func @ldmatrix_num_elements_f16_x4(%arg0: memref<128x128xf16, 3>) ->  vector<4x1xf16> {
  %c0  = arith.constant 0 : index
  // expected-error @+1 {{expected vector register shape[1] = 2}}
  %a = nvgpu.ldmatrix %arg0[%c0, %c0] {transpose = false, numTiles = 4 : i32} : memref<128x128xf16, 3> -> vector<4x1xf16>
  return %a : vector<4x1xf16>
}
// -----

func.func @ldmatrix_num_tiles_f16_x4(%arg0: memref<128x128xf16, 3>) ->  vector<2x2xf16> {
  %c0  = arith.constant 0 : index
  // expected-error @+1 {{expected vector register shape[0] and numTiles to match}}
  %a = nvgpu.ldmatrix %arg0[%c0, %c0] {transpose = false, numTiles = 4 : i32} : memref<128x128xf16, 3> -> vector<2x2xf16>
  return %a : vector<2x2xf16>
}
// -----

func.func @ldmatrix_num_tiles_f32_x4(%arg0: memref<128x128xf32, 3>) ->  vector<4x2xf32> {
  %c0  = arith.constant 0 : index
  // expected-error @+1 {{expected vector register shape[1] = 1}}
  %a = nvgpu.ldmatrix %arg0[%c0, %c0] {transpose = false, numTiles = 4 : i32} : memref<128x128xf32, 3> -> vector<4x2xf32>
  return %a : vector<4x2xf32>
}
// -----

func.func @ldmatrix_trans_f32_x4(%arg0: memref<128x128xf32, 3>) ->  vector<4x1xf32> {
  %c0  = arith.constant 0 : index
  // expected-error @+1 {{nvgpu.ldmatrix transpose works only at 16b granularity}}
  %a = nvgpu.ldmatrix %arg0[%c0, %c0] {transpose = true, numTiles = 4 : i32} : memref<128x128xf32, 3> -> vector<4x1xf32>
  return %a : vector<4x1xf32>
}
// -----

func.func @ldmatrix_type_x4(%arg0: memref<128x128xf32, 3>) ->  vector<4x2xf16> {
  %c0  = arith.constant 0 : index
  // expected-error @+1 {{'nvgpu.ldmatrix' op failed to verify that srcMemref and res have same element type}}
  %a = nvgpu.ldmatrix %arg0[%c0, %c0] {transpose = false, numTiles = 4 : i32} : memref<128x128xf32, 3> -> vector<4x2xf16>
  return %a : vector<4x2xf16>
}
// -----

func.func @m16n8k16_fp16_vector_shape_a(%arg0: vector<4x4xf16>, %arg1: vector<2x2xf16>, %arg2: vector<2x2xf16>) -> vector<2x2xf16> {
  // expected-error @+1 {{expected 256 warp-wide matrix A elements}}
  %d = nvgpu.mma.sync (%arg0, %arg1, %arg2) {mmaShape = [16, 8, 16]} : (vector<4x4xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
  return %d : vector<2x2xf16>
}
// -----

func.func @m16n8k16_fp16_vector_shape_b(%arg0: vector<4x2xf16>, %arg1: vector<2x4xf16>, %arg2: vector<2x2xf16>) -> vector<2x2xf16> {
  // expected-error @+1 {{expected 128 warp-wide matrix B elements}}
  %d = nvgpu.mma.sync (%arg0, %arg1, %arg2) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x4xf16>, vector<2x2xf16>) -> vector<2x2xf16>
  return %d : vector<2x2xf16>
}
// -----

func.func @m16n8k16_fp16_vector_shape_c(%arg0: vector<4x2xf16>, %arg1: vector<2x2xf16>, %arg2: vector<2x4xf16>) -> vector<2x4xf16> {
  // expected-error @+1 {{expected 128 warp-wide matrix C elements}}
  %d = nvgpu.mma.sync (%arg0, %arg1, %arg2) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x4xf16>) -> vector<2x4xf16>
  return %d : vector<2x4xf16>
}
// -----

func.func @m16n8k16_fp16_vector_shape_a_extended(%arg0: vector<2x4xf16>, %arg1: vector<2x2xf16>, %arg2: vector<2x2xf16>) -> vector<2x2xf16> {
  // expected-error @+1 {{expected matrix A to be shaped (4 x 2)}}
  %d = nvgpu.mma.sync (%arg0, %arg1, %arg2) {mmaShape = [16, 8, 16]} : (vector<2x4xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
  return %d : vector<2x2xf16>
}
// -----

func.func @m16n8k16_fp16_tf32Enabled(%arg0: vector<4x2xf16>, %arg1: vector<2x2xf16>, %arg2: vector<2x2xf16>) -> vector<2x2xf16> {
  // expected-error @+1 {{expected tf32 tensor cores only for F32 operands}}
  %d = nvgpu.mma.sync (%arg0, %arg1, %arg2) {mmaShape = [16, 8, 16], tf32Enabled} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
  return %d : vector<2x2xf16>
}
// -----

func.func @m16n8k8_fp32_vector_shape_a(%arg0: vector<4x2xf32>, %arg1: vector<2x1xf32>, %arg2: vector<2x2xf32>) -> vector<2x2xf32> {
  // expected-error @+1 {{expected 128 warp-wide matrix A elements}}
  %d = nvgpu.mma.sync (%arg0, %arg1, %arg2) {mmaShape = [16, 8, 8]} : (vector<4x2xf32>, vector<2x1xf32>, vector<2x2xf32>) -> vector<2x2xf32>
  return %d : vector<2x2xf32>
}
// -----

func.func @m16n8k8_fp32_vector_shape_a_extended(%arg0: vector<1x4xf32>, %arg1: vector<2x1xf32>, %arg2: vector<2x2xf32>) -> vector<2x2xf32> {
  // expected-error @+1 {{expected matrix A to be shaped (4 x 1)}}
  %d = nvgpu.mma.sync (%arg0, %arg1, %arg2) {mmaShape = [16, 8, 8]} : (vector<1x4xf32>, vector<2x1xf32>, vector<2x2xf32>) -> vector<2x2xf32>
  return %d : vector<2x2xf32>
}
// -----

func.func @m8n8k4_fp64_vector_shape_a(%arg0: vector<1x2xf64>, %arg1: vector<1x1xf64>, %arg2: vector<1x2xf64>) -> vector<1x2xf64> {
  // expected-error @+1 {{expected 32 warp-wide matrix A elements}}
  %d = nvgpu.mma.sync (%arg0, %arg1, %arg2) {mmaShape = [8, 8, 4]} : (vector<1x2xf64>, vector<1x1xf64>, vector<1x2xf64>) -> vector<1x2xf64>
  return %d : vector<1x2xf64>
}
// -----

func.func @m8n8k4_fp64_vector_shape_c_extended(%arg0: vector<1x1xf64>, %arg1: vector<1x1xf64>, %arg2: vector<2x1xf64>) -> vector<2x1xf64> {
  // expected-error @+1 {{expected matrix C to be shaped (1 x 2)}}
  %d = nvgpu.mma.sync (%arg0, %arg1, %arg2) {mmaShape = [8, 8, 4]} : (vector<1x1xf64>, vector<1x1xf64>, vector<2x1xf64>) -> vector<2x1xf64>
  return %d : vector<2x1xf64>
}
// -----

func.func @m16n8k32_int8_vector_shape_b(%arg0: vector<4x4xi8>, %arg1: vector<4x4xi8>, %arg2: vector<2x2xi32>) -> vector<2x2xi32> {
  // expected-error @+1 {{expected 256 warp-wide matrix B elements}}
  %d = nvgpu.mma.sync (%arg0, %arg1, %arg2) {mmaShape = [16, 8, 32]} : (vector<4x4xi8>, vector<4x4xi8>, vector<2x2xi32>) -> vector<2x2xi32>
  return %d : vector<2x2xi32>
}
// -----

func.func @m16n8k32_int32_datatype(%arg0: vector<4x4xi32>, %arg1: vector<2x4xi8>, %arg2: vector<2x2xi32>) -> vector<2x2xi32> {
  // expected-error @+1 {{op failed to verify that matrixA and matrixB have same element type}}
  %d = nvgpu.mma.sync (%arg0, %arg1, %arg2) {mmaShape = [16, 8, 32]} : (vector<4x4xi32>, vector<2x4xi8>, vector<2x2xi32>) -> vector<2x2xi32>
  return %d : vector<2x2xi32>
}
// -----

func.func @async_cp_memory_space(%dst : memref<16xf32>, %src : memref<16xf32>, %i : index) -> () {
  // expected-error @below {{destination memref must have a memory space attribute of IntegerAttr(3) or gpu::AddressSpaceAttr(Workgroup)}}
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
// -----

// 42 is never the answer!
func.func @mma_sp_sync_f16_16816(%arg0: vector<2x2xf16>,
                                 %arg1: vector<2x2xf16>,
                                 %arg2: vector<2x2xf16>,
                                 %arg3: vector<2xi16>) -> vector<2x2xf16> {
  // expected-error @+1 {{'nvgpu.mma.sp.sync' op sparsity selector should be 0 or 1}}
  %d = nvgpu.mma.sp.sync(%arg0, %arg1, %arg2) metadata(%arg3) {mmaShape = [16, 8, 16], sparsitySelector = 42 : i32} :
       (vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
  return %d : vector<2x2xf16>
}

// -----

func.func @async_cp_zfill_f32_align1(
  %src: memref<128x128xf32>, %dst: memref<3x16x128xf32, 3>, %i : index, %srcElements : index) {
    // expected-error @+1 {{'nvgpu.device_async_copy' op bypassL1 does not satify alignment for 'memref<3x16x128xf32, 3>' with destination element 1. Unset bypassL1, or set destination element to 4}}
  %0 = nvgpu.device_async_copy %src[%i, %i], %dst[%i, %i, %i], 1, %srcElements {bypassL1} : memref<128x128xf32> to memref<3x16x128xf32, 3>
  return
}

// -----

func.func @async_cp_size_invalid_f32(
  %src: memref<128x128xf32>, %dst: memref<3x16x128xf32, 3>, %i : index) {
    // expected-error @+1 {{Requested copy elements is 3 with width 32. But copy elements could be one of 1, 2, 4.}}
  %0 = nvgpu.device_async_copy %src[%i, %i], %dst[%i, %i, %i], 3: memref<128x128xf32> to memref<3x16x128xf32, 3>
  return
}

// -----

func.func @async_cp_size_invalid_f16(
  %src: memref<128x128xf16>, %dst: memref<3x16x128xf16, 3>, %i : index) {
    // expected-error @+1 {{Requested copy elements is 3 with width 16. But copy elements could be one of 2, 4, 8.}}
  %0 = nvgpu.device_async_copy %src[%i, %i], %dst[%i, %i, %i], 3: memref<128x128xf16> to memref<3x16x128xf16, 3>
  return
}

// -----

func.func @async_cp_size_invalid_f64(
  %src: memref<128x128xf64>, %dst: memref<3x16x128xf64, 3>, %i : index) {
    // expected-error @+1 {{Requested copy elements is 3 with width 64. But copy elements could be one of 1, 2.}}
  %0 = nvgpu.device_async_copy %src[%i, %i], %dst[%i, %i, %i], 3: memref<128x128xf64> to memref<3x16x128xf64, 3>
  return
}

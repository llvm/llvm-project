// RUN: mlir-opt %s -split-input-file -pass-pipeline="builtin.module(func.func(test-fold-arith-extf-into-vector-contract-patterns,convert-vector-to-gpu{use-nvgpu=true},cse))" | FileCheck %s

//###############################################################################################
// FP16 input, F32 accumulation row-row-row (ldmatrix x4 for matrixA and ldmatrix x4 for matrixB)
//###############################################################################################

#map0 = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @m16n8k16_mmasync16816_f16_f16_f32_row_row_row
func.func @m16n8k16_mmasync16816_f16_f16_f32_row_row_row(%arg0: memref<42x32xf16, #gpu.address_space<workgroup>>, %arg1: memref<32x64xf16, #gpu.address_space<workgroup>>, %arg2: memref<42x64xf32, #gpu.address_space<workgroup>>) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %cst_f16 = arith.constant 0.000000e+00 : f16
  %cst_f32 = arith.constant 0.000000e+00 : f32
  
  // CHECK-DAG: nvgpu.ldmatrix %arg0[%{{.*}}, %{{.*}}] {numTiles = 4 : i32, transpose = false}
  %A = vector.transfer_read %arg0[%c0, %c0], %cst_f16 {in_bounds = [true, true]} : memref<42x32xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
  %A_f32 = arith.extf %A : vector<16x16xf16> to vector<16x16xf32>
  

  // CHECK-DAG: nvgpu.ldmatrix %arg1[%{{.*}}, %{{.*}}] {numTiles = 4 : i32, transpose = true}
  %B = vector.transfer_read %arg1[%c0, %c0], %cst_f16 {permutation_map = #map0, in_bounds = [true, true]} : memref<32x64xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
  %C = vector.transfer_read %arg2[%c0, %c0], %cst_f32 {in_bounds = [true, true]} : memref<42x64xf32, #gpu.address_space<workgroup>>, vector<16x16xf32>

  %B0 = vector.extract_strided_slice %B {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
  %B0_f32 = arith.extf %B0 : vector<8x16xf16> to vector<8x16xf32>
  %C0 = vector.extract_strided_slice %C {offsets = [0, 0], sizes = [16, 8], strides = [1, 1]} : vector<16x16xf32> to vector<16x8xf32>
  
  // CHECK-DAG: nvgpu.mma.sync({{.*}}) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
  %D0 = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A_f32, %B0_f32, %C0 : vector<16x16xf32>, vector<8x16xf32> into vector<16x8xf32>
  vector.transfer_write %D0, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x8xf32>, memref<42x64xf32, #gpu.address_space<workgroup>>


  %B1 = vector.extract_strided_slice %B {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
  %B1_f32 = arith.extf %B1 : vector<8x16xf16> to vector<8x16xf32>
  %C1 = vector.extract_strided_slice %C {offsets = [0, 8], sizes = [16, 8], strides = [1, 1]} : vector<16x16xf32> to vector<16x8xf32>

  // CHECK-DAG: nvgpu.mma.sync({{.*}}) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
  %D1 = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A_f32, %B1_f32, %C1 : vector<16x16xf32>, vector<8x16xf32> into vector<16x8xf32>
  vector.transfer_write %D1, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x8xf32>, memref<42x64xf32, #gpu.address_space<workgroup>>

  return
}

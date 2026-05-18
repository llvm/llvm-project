// RUN: mlir-opt --fold-memref-alias-ops --split-input-file %s | FileCheck %s

#gpu_lds_addrspace = #gpu.address_space<workgroup>

// CHECK: func @test_subview_folding
// CHECK-SAME: %[[ARG0:.*]]: index, %[[ARG1:.*]]: index
func.func @test_subview_folding(%offset_i: index, %offset_j: index) {
  // CHECK-DAG: %[[LOCAL:.*]] = memref.alloc() : memref<64x64xf16, #gpu.address_space<workgroup>>
  // CHECK-DAG: %[[MEM:.*]] = memref.alloc() : memref<64x128xf16>
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: amdgpu.gather_to_lds %[[MEM]][%[[ARG0]], %[[ARG1]]], %[[LOCAL]][%[[C0]], %[[C0]]]
  // CHECK-SAME: vector<8xf16>, memref<64x128xf16>, memref<64x64xf16, #gpu.address_space<workgroup>>

  %alloc = memref.alloc() : memref<64x64xf16, #gpu_lds_addrspace>
  %mem = memref.alloc() : memref<64x128xf16>
  %subview = memref.subview %mem[0, 0][32, 64][1, 1] : memref<64x128xf16> to memref<32x64xf16, strided<[128, 1]>>
  %c0 = arith.constant 0 : index
  amdgpu.gather_to_lds %subview[%offset_i, %offset_j], %alloc[%c0, %c0]
    : vector<8xf16>, memref<32x64xf16, strided<[128, 1]>>, memref<64x64xf16, #gpu_lds_addrspace>
  func.return
}

// -----

#gpu_lds_addrspace = #gpu.address_space<workgroup>

// CHECK: #[[MAP:.*]] = affine_map<()[s0] -> (s0 + 32)>
// CHECK: #[[MAP1:.*]] = affine_map<()[s0] -> (s0 + 64)>

// CHECK: func @subview_folding_offset
// CHECK-SAME: %[[ARG0:.*]]: index, %[[ARG1:.*]]: index
func.func @subview_folding_offset(%offset_i: index, %offset_j: index) {
  // CHECK-DAG: %[[LOCAL:.*]] = memref.alloc() : memref<64x64xf16, #gpu.address_space<workgroup>>
  // CHECK-DAG: %[[MEM:.*]] = memref.alloc() : memref<64x128xf16>
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[IDX0:.*]] = affine.apply #[[MAP]]()[%[[ARG0]]]
  // CHECK: %[[IDX1:.*]] = affine.apply #[[MAP1]]()[%[[ARG1]]]
  // CHECK: amdgpu.gather_to_lds %[[MEM]][%[[IDX0]], %[[IDX1]]], %[[LOCAL]][%[[C0]], %[[C0]]]
  // CHECK-SAME: vector<8xf16>, memref<64x128xf16>, memref<64x64xf16, #gpu.address_space<workgroup>>

  %alloc = memref.alloc() : memref<64x64xf16, #gpu_lds_addrspace>
  %mem = memref.alloc() : memref<64x128xf16>
  %subview = memref.subview %mem[32, 64][32, 64][1, 1] : memref<64x128xf16> to memref<32x64xf16, strided<[128, 1], offset: 4160>>
  %c0 = arith.constant 0 : index
  amdgpu.gather_to_lds %subview[%offset_i, %offset_j], %alloc[%c0, %c0]
    : vector<8xf16>, memref<32x64xf16, strided<[128, 1], offset: 4160>>, memref<64x64xf16, #gpu_lds_addrspace>
  func.return
}

// -----

#gpu_lds_addrspace = #gpu.address_space<workgroup>

// CHECK: func @test_expand_shape
// CHECK-SAME: %[[ARG0:.*]]: index, %[[ARG1:.*]]: index
func.func @test_expand_shape(%offset_i: index, %offset_j: index) {
  // CHECK-DAG: %[[LOCAL:.*]] = memref.alloc() : memref<4096xf16, #gpu.address_space<workgroup>>
  // CHECK-DAG: %[[MEM:.*]] = memref.alloc() : memref<8192xf16>
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[IDXM:.*]] = affine.linearize_index disjoint [%[[ARG0]], %[[ARG1]]] by (64, 128) : index
  // CHECK: amdgpu.gather_to_lds %[[MEM]][%[[IDXM]]], %[[LOCAL]][%[[C0]]]
  // CHECK-SAME: vector<8xf16>, memref<8192xf16>, memref<4096xf16, #gpu.address_space<workgroup>>

  %alloc = memref.alloc() : memref<4096xf16, #gpu_lds_addrspace>
  %mem = memref.alloc() : memref<8192xf16>
  %expand_mem = memref.expand_shape %mem [[0, 1]] output_shape [64, 128] : memref<8192xf16> into memref<64x128xf16>
  %expand_alloc = memref.expand_shape %alloc [[0, 1]] output_shape [64, 64] : memref<4096xf16, #gpu_lds_addrspace> into memref<64x64xf16, #gpu_lds_addrspace>
  %c0 = arith.constant 0 : index
  amdgpu.gather_to_lds %expand_mem[%offset_i, %offset_j], %expand_alloc[%c0, %c0]
    : vector<8xf16>, memref<64x128xf16>, memref<64x64xf16, #gpu_lds_addrspace>
  func.return
}

// -----

#gpu_lds_addrspace = #gpu.address_space<workgroup>

// CHECK: func @test_collapse_shape
// CHECK-SAME: %[[ARG0:.*]]: index, %[[ARG1:.*]]: index
func.func @test_collapse_shape(%offset_i: index, %offset_j: index) {
  // CHECK-DAG: %[[LOCAL:.*]] = memref.alloc() : memref<64x64xf16, #gpu.address_space<workgroup>>
  // CHECK-DAG: %[[MEM:.*]] = memref.alloc() : memref<64x128xf16>
  // CHECK: %[[INDICES_MEM:.*]]:2 = affine.delinearize_index %[[ARG0]] into (64, 128) : index, index
  // CHECK: %[[INDICES_LDS:.*]]:2 = affine.delinearize_index %[[ARG1]] into (64, 64) : index, index
  // CHECK: amdgpu.gather_to_lds %[[MEM]][%[[INDICES_MEM]]#0, %[[INDICES_MEM]]#1], %[[LOCAL]][%[[INDICES_LDS]]#0, %[[INDICES_LDS]]#1]
  // CHECK-SAME: vector<8xf16>, memref<64x128xf16>, memref<64x64xf16, #gpu.address_space<workgroup>>

  %alloc = memref.alloc() : memref<64x64xf16, #gpu_lds_addrspace>
  %collapse_alloc = memref.collapse_shape %alloc [[0, 1]] : memref<64x64xf16, #gpu_lds_addrspace> into memref<4096xf16, #gpu_lds_addrspace>
  %mem = memref.alloc() : memref<64x128xf16>
  %collapse_mem = memref.collapse_shape %mem [[0, 1]] : memref<64x128xf16> into memref<8192xf16>
  %c0 = arith.constant 0 : index
  amdgpu.gather_to_lds %collapse_mem[%offset_i], %collapse_alloc[%offset_j]
    : vector<8xf16>, memref<8192xf16>, memref<4096xf16, #gpu_lds_addrspace>
  func.return
}


// -----

#gpu_lds_addrspace = #gpu.address_space<workgroup>


// CHECK: func @test_expand_shape_src_raw_buffer
// CHECK-SAME: %[[ARG0:.*]]: memref<8192xf16, #amdgpu.address_space<fat_raw_buffer>>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index
func.func @test_expand_shape_src_raw_buffer(%mem : memref<8192xf16, #amdgpu.address_space<fat_raw_buffer>>, %offset_i: index, %offset_j: index) {
  // CHECK-DAG: %[[LOCAL:.*]] = memref.alloc() : memref<4096xf16, #gpu.address_space<workgroup>>
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[IDXM:.*]] = affine.linearize_index [%[[ARG1]], %[[ARG2]]] by (64, 128) : index
  // CHECK: amdgpu.gather_to_lds %[[ARG0]][%[[IDXM]]], %[[LOCAL]][%[[C0]]]
  // CHECK-SAME: vector<8xf16>, memref<8192xf16, #amdgpu.address_space<fat_raw_buffer>>, memref<4096xf16, #gpu.address_space<workgroup>>

  %alloc = memref.alloc() : memref<4096xf16, #gpu_lds_addrspace>
  %expand_mem = memref.expand_shape %mem [[0, 1]] output_shape [64, 128] : memref<8192xf16, #amdgpu.address_space<fat_raw_buffer>> into memref<64x128xf16, #amdgpu.address_space<fat_raw_buffer>>

  %c0 = arith.constant 0 : index
  amdgpu.gather_to_lds %expand_mem[%offset_i, %offset_j], %alloc[%c0]
    : vector<8xf16>, memref<64x128xf16, #amdgpu.address_space<fat_raw_buffer>>, memref<4096xf16, #gpu_lds_addrspace>
  func.return
}

// -----

#gpu_lds_addrspace = #gpu.address_space<workgroup>

// CHECK: func @test_src_raw_buffer_collapse_shape
// CHECK-SAME: %[[MEM:.*]]: memref<64x128xf16, #amdgpu.address_space<fat_raw_buffer>>, %[[IDX:.*]]: index
func.func @test_src_raw_buffer_collapse_shape(%mem : memref<64x128xf16, #amdgpu.address_space<fat_raw_buffer>>, %offset: index) {
  // CHECK-DAG: %[[LOCAL:.*]] = memref.alloc() : memref<4096xf16, #gpu.address_space<workgroup>>
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[INDICES:.*]]:2 = affine.delinearize_index %[[IDX]] into (128) : index, index
  // CHECK: amdgpu.gather_to_lds %[[MEM]][%[[INDICES]]#0, %[[INDICES]]#1], %[[LOCAL]][%[[C0]]]
  // CHECK-SAME: vector<8xf16>, memref<64x128xf16, #amdgpu.address_space<fat_raw_buffer>>, memref<4096xf16, #gpu.address_space<workgroup>>

  %alloc = memref.alloc() : memref<4096xf16, #gpu_lds_addrspace>
  %collapse_mem = memref.collapse_shape %mem [[0, 1]]
    : memref<64x128xf16, #amdgpu.address_space<fat_raw_buffer>>
    into memref<8192xf16, #amdgpu.address_space<fat_raw_buffer>>
  %c0 = arith.constant 0 : index
  amdgpu.gather_to_lds %collapse_mem[%offset], %alloc[%c0]
    : vector<8xf16>, memref<8192xf16, #amdgpu.address_space<fat_raw_buffer>>,
      memref<4096xf16, #gpu_lds_addrspace>
  func.return
}

// -----

#gpu_lds_addrspace = #gpu.address_space<workgroup>

// CHECK: func @test_expand_shape_dst_only
// CHECK-SAME: %[[ARG0:.*]]: index, %[[ARG1:.*]]: index
func.func @test_expand_shape_dst_only(%offset_i: index, %offset_j: index) {
  // CHECK-DAG: %[[LOCAL:.*]] = memref.alloc() : memref<4096xf16, #gpu.address_space<workgroup>>
  // CHECK-DAG: %[[MEM:.*]] = memref.alloc() : memref<8192xf16>
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[IDX_LDS:.*]] = affine.linearize_index disjoint [%[[ARG1]], %[[C0]]] by (64, 64) : index
  // CHECK: amdgpu.gather_to_lds %[[MEM]][%[[ARG0]]], %[[LOCAL]][%[[IDX_LDS]]]
  // CHECK-SAME: vector<8xf16>, memref<8192xf16>, memref<4096xf16, #gpu.address_space<workgroup>>

  %alloc = memref.alloc() : memref<4096xf16, #gpu_lds_addrspace>
  %mem = memref.alloc() : memref<8192xf16>
  %expand_alloc = memref.expand_shape %alloc [[0, 1]] output_shape [64, 64] : memref<4096xf16, #gpu_lds_addrspace> into memref<64x64xf16, #gpu_lds_addrspace>

  %c0 = arith.constant 0 : index
  amdgpu.gather_to_lds %mem[%offset_i], %expand_alloc[%offset_j, %c0]
    : vector<8xf16>, memref<8192xf16>, memref<64x64xf16, #gpu_lds_addrspace>
  func.return
}

// -----

#gpu_lds_addrspace = #gpu.address_space<workgroup>

// CHECK: func @test_nop
// CHECK-SAME: %[[ARG0:.*]]: memref<8192xf16, #amdgpu.address_space<fat_raw_buffer>>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index
func.func @test_nop(%mem : memref<8192xf16, #amdgpu.address_space<fat_raw_buffer>>, %offset_i: index, %offset_j: index) {
  // CHECK-DAG: %[[LOCAL:.*]] = memref.alloc() : memref<4096xf16, #gpu.address_space<workgroup>>
  // CHECK: amdgpu.gather_to_lds %[[ARG0]][%[[ARG1]]], %[[LOCAL]][%[[ARG2]]]
  // CHECK-SAME: vector<8xf16>, memref<8192xf16, #amdgpu.address_space<fat_raw_buffer>>, memref<4096xf16, #gpu.address_space<workgroup>>

  %alloc = memref.alloc() : memref<4096xf16, #gpu_lds_addrspace>
  amdgpu.gather_to_lds %mem[%offset_i], %alloc[%offset_j]
    : vector<8xf16>, memref<8192xf16, #amdgpu.address_space<fat_raw_buffer>>, memref<4096xf16, #gpu_lds_addrspace>
  func.return
}

// -----

#gpu_lds_addrspace = #gpu.address_space<workgroup>

// CHECK: func @test_async_flag_preserved
// CHECK-SAME: %[[ARG0:.*]]: index, %[[ARG1:.*]]: index
func.func @test_async_flag_preserved(%offset_i: index, %offset_j: index) {
  // CHECK-DAG: %[[LOCAL:.*]] = memref.alloc() : memref<64x64xf16, #gpu.address_space<workgroup>>
  // CHECK-DAG: %[[MEM:.*]] = memref.alloc() : memref<64x128xf16>
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: amdgpu.gather_to_lds async %[[MEM]][%[[ARG0]], %[[ARG1]]], %[[LOCAL]][%[[C0]], %[[C0]]]
  // CHECK-SAME: vector<8xf16>, memref<64x128xf16>, memref<64x64xf16, #gpu.address_space<workgroup>>

  %alloc = memref.alloc() : memref<64x64xf16, #gpu_lds_addrspace>
  %mem = memref.alloc() : memref<64x128xf16>
  %subview = memref.subview %mem[0, 0][32, 64][1, 1] : memref<64x128xf16> to memref<32x64xf16, strided<[128, 1]>>
  %c0 = arith.constant 0 : index
  amdgpu.gather_to_lds async %subview[%offset_i, %offset_j], %alloc[%c0, %c0]
    : vector<8xf16>, memref<32x64xf16, strided<[128, 1]>>, memref<64x64xf16, #gpu_lds_addrspace>
  func.return
}

// -----

#gpu_wg = #gpu.address_space<workgroup>

// CHECK: func @test_transpose_load_subview
// CHECK-SAME: %[[ARG0:.*]]: index, %[[ARG1:.*]]: index
func.func @test_transpose_load_subview(%offset_i: index, %offset_j: index) -> vector<4xf16> {
  // CHECK-DAG: %[[ALLOC:.*]] = memref.alloc() : memref<64x128xf16, #gpu.address_space<workgroup>>
  // CHECK: amdgpu.transpose_load %[[ALLOC]][%[[ARG0]], %[[ARG1]]]
  // CHECK-SAME: memref<64x128xf16, #gpu.address_space<workgroup>> -> vector<4xf16>

  %alloc = memref.alloc() : memref<64x128xf16, #gpu_wg>
  %subview = memref.subview %alloc[0, 0][32, 64][1, 1]
    : memref<64x128xf16, #gpu_wg> to memref<32x64xf16, strided<[128, 1]>, #gpu_wg>
  %result = amdgpu.transpose_load %subview[%offset_i, %offset_j]
    : memref<32x64xf16, strided<[128, 1]>, #gpu_wg> -> vector<4xf16>
  return %result : vector<4xf16>
}

// -----

#gpu_wg = #gpu.address_space<workgroup>

// CHECK: #[[MAP:.*]] = affine_map<()[s0] -> (s0 + 32)>
// CHECK: #[[MAP1:.*]] = affine_map<()[s0] -> (s0 + 64)>

// CHECK: func @test_transpose_load_subview_offset
// CHECK-SAME: %[[ARG0:.*]]: index, %[[ARG1:.*]]: index
func.func @test_transpose_load_subview_offset(%offset_i: index, %offset_j: index) -> vector<4xf16> {
  // CHECK-DAG: %[[ALLOC:.*]] = memref.alloc() : memref<64x128xf16, #gpu.address_space<workgroup>>
  // CHECK: %[[IDX0:.*]] = affine.apply #[[MAP]]()[%[[ARG0]]]
  // CHECK: %[[IDX1:.*]] = affine.apply #[[MAP1]]()[%[[ARG1]]]
  // CHECK: amdgpu.transpose_load %[[ALLOC]][%[[IDX0]], %[[IDX1]]]
  // CHECK-SAME: memref<64x128xf16, #gpu.address_space<workgroup>> -> vector<4xf16>

  %alloc = memref.alloc() : memref<64x128xf16, #gpu_wg>
  %subview = memref.subview %alloc[32, 64][32, 64][1, 1]
    : memref<64x128xf16, #gpu_wg>
    to memref<32x64xf16, strided<[128, 1], offset: 4160>, #gpu_wg>
  %result = amdgpu.transpose_load %subview[%offset_i, %offset_j]
    : memref<32x64xf16, strided<[128, 1], offset: 4160>, #gpu_wg> -> vector<4xf16>
  return %result : vector<4xf16>
}

// -----

#gpu_wg = #gpu.address_space<workgroup>

// CHECK: func @test_transpose_load_expand_shape
// CHECK-SAME: %[[ARG0:.*]]: index, %[[ARG1:.*]]: index
func.func @test_transpose_load_expand_shape(%offset_i: index, %offset_j: index) -> vector<4xf16> {
  // CHECK-DAG: %[[ALLOC:.*]] = memref.alloc() : memref<4096xf16, #gpu.address_space<workgroup>>
  // CHECK: %[[IDX:.*]] = affine.linearize_index disjoint [%[[ARG0]], %[[ARG1]]] by (32, 128) : index
  // CHECK: amdgpu.transpose_load %[[ALLOC]][%[[IDX]]]
  // CHECK-SAME: memref<4096xf16, #gpu.address_space<workgroup>> -> vector<4xf16>

  %alloc = memref.alloc() : memref<4096xf16, #gpu_wg>
  %expand = memref.expand_shape %alloc [[0, 1]] output_shape [32, 128]
    : memref<4096xf16, #gpu_wg> into memref<32x128xf16, #gpu_wg>
  %result = amdgpu.transpose_load %expand[%offset_i, %offset_j]
    : memref<32x128xf16, #gpu_wg> -> vector<4xf16>
  return %result : vector<4xf16>
}

// -----

#gpu_wg = #gpu.address_space<workgroup>

// CHECK: func @test_transpose_load_collapse_shape
// CHECK-SAME: %[[ARG0:.*]]: index
func.func @test_transpose_load_collapse_shape(%offset_i: index) -> vector<4xf16> {
  // CHECK-DAG: %[[ALLOC:.*]] = memref.alloc() : memref<32x128xf16, #gpu.address_space<workgroup>>
  // CHECK: %[[INDICES:.*]]:2 = affine.delinearize_index %[[ARG0]] into (32, 128) : index, index
  // CHECK: amdgpu.transpose_load %[[ALLOC]][%[[INDICES]]#0, %[[INDICES]]#1]
  // CHECK-SAME: memref<32x128xf16, #gpu.address_space<workgroup>> -> vector<4xf16>

  %alloc = memref.alloc() : memref<32x128xf16, #gpu_wg>
  %collapse = memref.collapse_shape %alloc [[0, 1]]
    : memref<32x128xf16, #gpu_wg> into memref<4096xf16, #gpu_wg>
  %result = amdgpu.transpose_load %collapse[%offset_i]
    : memref<4096xf16, #gpu_wg> -> vector<4xf16>
  return %result : vector<4xf16>
}

// -----

#gpu_wg = #gpu.address_space<workgroup>

// CHECK: func @test_transpose_load_nop
// CHECK-SAME: %[[ARG0:.*]]: index, %[[ARG1:.*]]: index
func.func @test_transpose_load_nop(%offset_i: index, %offset_j: index) -> vector<4xf16> {
  // CHECK-DAG: %[[ALLOC:.*]] = memref.alloc() : memref<32x128xf16, #gpu.address_space<workgroup>>
  // CHECK: amdgpu.transpose_load %[[ALLOC]][%[[ARG0]], %[[ARG1]]]
  // CHECK-SAME: memref<32x128xf16, #gpu.address_space<workgroup>> -> vector<4xf16>
  // CHECK-NOT: subview
  // CHECK-NOT: expand_shape

  %alloc = memref.alloc() : memref<32x128xf16, #gpu_wg>
  %result = amdgpu.transpose_load %alloc[%offset_i, %offset_j]
    : memref<32x128xf16, #gpu_wg> -> vector<4xf16>
  return %result : vector<4xf16>
}

// -----

#gpu_lds_addrspace = #gpu.address_space<workgroup>
#gpu_global_addrspace = #gpu.address_space<global>

// CHECK: func @test_make_dma_base_subview
// CHECK-SAME: %[[MEM:.*]]: memref<64x128xf16, #gpu.address_space<global>>, %[[LDS:.*]]: memref<64x64xf16, #gpu.address_space<workgroup>>, %[[GLOBAL_I:.*]]: index, %[[GLOBAL_J:.*]]: index, %[[LDS_I:.*]]: index, %[[LDS_J:.*]]: index
func.func @test_make_dma_base_subview(%mem: memref<64x128xf16, #gpu_global_addrspace>, %lds: memref<64x64xf16, #gpu_lds_addrspace>, %global_i: index, %global_j: index, %lds_i: index, %lds_j: index) -> !amdgpu.tdm_base<f16> {
  // CHECK: %[[BASE:.*]] = amdgpu.make_dma_base %[[MEM]][%[[GLOBAL_I]], %[[GLOBAL_J]]], %[[LDS]][%[[LDS_I]], %[[LDS_J]]]
  // CHECK-SAME: memref<64x128xf16, #gpu.address_space<global>>, memref<64x64xf16, #gpu.address_space<workgroup>> -> !amdgpu.tdm_base<f16>
  // CHECK: return %[[BASE]] : !amdgpu.tdm_base<f16>

  %subview = memref.subview %mem[0, 0][32, 64][1, 1] : memref<64x128xf16, #gpu_global_addrspace> to memref<32x64xf16, strided<[128, 1]>, #gpu_global_addrspace>
  %base = amdgpu.make_dma_base %subview[%global_i, %global_j], %lds[%lds_i, %lds_j]
    : memref<32x64xf16, strided<[128, 1]>, #gpu_global_addrspace>, memref<64x64xf16, #gpu_lds_addrspace> -> !amdgpu.tdm_base<f16>
  func.return %base : !amdgpu.tdm_base<f16>
}

// -----

#gpu_lds_addrspace = #gpu.address_space<workgroup>
#gpu_global_addrspace = #gpu.address_space<global>

// CHECK: func @test_make_dma_base_expand_shape
// CHECK-SAME: %[[MEM:.*]]: memref<64x128xf16, #gpu.address_space<global>>, %[[LDS:.*]]: memref<4096xf16, #gpu.address_space<workgroup>>, %[[GLOBAL_I:.*]]: index, %[[GLOBAL_J:.*]]: index, %[[LDS_I:.*]]: index, %[[LDS_J:.*]]: index
func.func @test_make_dma_base_expand_shape(%mem: memref<64x128xf16, #gpu_global_addrspace>, %lds: memref<4096xf16, #gpu_lds_addrspace>, %global_i: index, %global_j: index, %lds_i: index, %lds_j: index) -> !amdgpu.tdm_base<f16> {
  // CHECK: %[[IDX:.*]] = affine.linearize_index disjoint [%[[LDS_I]], %[[LDS_J]]] by (64, 64) : index
  // CHECK: %[[BASE:.*]] = amdgpu.make_dma_base %[[MEM]][%[[GLOBAL_I]], %[[GLOBAL_J]]], %[[LDS]][%[[IDX]]]
  // CHECK-SAME: memref<64x128xf16, #gpu.address_space<global>>, memref<4096xf16, #gpu.address_space<workgroup>> -> !amdgpu.tdm_base<f16>
  // CHECK: return %[[BASE]] : !amdgpu.tdm_base<f16>

  %expand_lds = memref.expand_shape %lds [[0, 1]] output_shape [64, 64] : memref<4096xf16, #gpu_lds_addrspace> into memref<64x64xf16, #gpu_lds_addrspace>
  %base = amdgpu.make_dma_base %mem[%global_i, %global_j], %expand_lds[%lds_i, %lds_j]
    : memref<64x128xf16, #gpu_global_addrspace>, memref<64x64xf16, #gpu_lds_addrspace> -> !amdgpu.tdm_base<f16>
  func.return %base : !amdgpu.tdm_base<f16>
}

// -----

#gpu_lds_addrspace = #gpu.address_space<workgroup>
#gpu_global_addrspace = #gpu.address_space<global>

// CHECK: func @test_make_gather_dma_base_subview
// CHECK-SAME: %[[MEM:.*]]: memref<64x128xf16, #gpu.address_space<global>>, %[[LDS:.*]]: memref<64x64xf16, #gpu.address_space<workgroup>>, %[[GLOBAL_I:.*]]: index, %[[GLOBAL_J:.*]]: index, %[[LDS_I:.*]]: index, %[[LDS_J:.*]]: index
func.func @test_make_gather_dma_base_subview(%mem: memref<64x128xf16, #gpu_global_addrspace>, %lds: memref<64x64xf16, #gpu_lds_addrspace>, %global_i: index, %global_j: index, %lds_i: index, %lds_j: index) -> !amdgpu.tdm_gather_base<f16, i16> {
  // CHECK: %[[BASE:.*]] = amdgpu.make_gather_dma_base %[[MEM]][%[[GLOBAL_I]], %[[GLOBAL_J]]], %[[LDS]][%[[LDS_I]], %[[LDS_J]]]
  // CHECK-SAME: memref<64x128xf16, #gpu.address_space<global>>, memref<64x64xf16, #gpu.address_space<workgroup>> -> !amdgpu.tdm_gather_base<f16, i16>
  // CHECK: return %[[BASE]] : !amdgpu.tdm_gather_base<f16, i16>

  %subview = memref.subview %mem[0, 0][32, 64][1, 1] : memref<64x128xf16, #gpu_global_addrspace> to memref<32x64xf16, strided<[128, 1]>, #gpu_global_addrspace>
  %base = amdgpu.make_gather_dma_base %subview[%global_i, %global_j], %lds[%lds_i, %lds_j]
    : memref<32x64xf16, strided<[128, 1]>, #gpu_global_addrspace>, memref<64x64xf16, #gpu_lds_addrspace> -> !amdgpu.tdm_gather_base<f16, i16>
  func.return %base : !amdgpu.tdm_gather_base<f16, i16>
}

// -----

#gpu_lds_addrspace = #gpu.address_space<workgroup>
#gpu_global_addrspace = #gpu.address_space<global>

// CHECK: func @test_make_gather_dma_base_collapse_shape
// CHECK-SAME: %[[MEM:.*]]: memref<64x128xf16, #gpu.address_space<global>>, %[[LDS:.*]]: memref<64x64xf16, #gpu.address_space<workgroup>>, %[[GLOBAL_I:.*]]: index, %[[GLOBAL_J:.*]]: index, %[[LDS_IDX:.*]]: index
func.func @test_make_gather_dma_base_collapse_shape(%mem: memref<64x128xf16, #gpu_global_addrspace>, %lds: memref<64x64xf16, #gpu_lds_addrspace>, %global_i: index, %global_j: index, %lds_idx: index) -> !amdgpu.tdm_gather_base<f16, i16> {
  // CHECK: %[[INDICES:.*]]:2 = affine.delinearize_index %[[LDS_IDX]] into (64, 64) : index, index
  // CHECK: %[[BASE:.*]] = amdgpu.make_gather_dma_base %[[MEM]][%[[GLOBAL_I]], %[[GLOBAL_J]]], %[[LDS]][%[[INDICES]]#0, %[[INDICES]]#1]
  // CHECK-SAME: memref<64x128xf16, #gpu.address_space<global>>, memref<64x64xf16, #gpu.address_space<workgroup>> -> !amdgpu.tdm_gather_base<f16, i16>
  // CHECK: return %[[BASE]] : !amdgpu.tdm_gather_base<f16, i16>

  %collapse_lds = memref.collapse_shape %lds [[0, 1]] : memref<64x64xf16, #gpu_lds_addrspace> into memref<4096xf16, #gpu_lds_addrspace>
  %base = amdgpu.make_gather_dma_base %mem[%global_i, %global_j], %collapse_lds[%lds_idx]
    : memref<64x128xf16, #gpu_global_addrspace>, memref<4096xf16, #gpu_lds_addrspace> -> !amdgpu.tdm_gather_base<f16, i16>
  func.return %base : !amdgpu.tdm_gather_base<f16, i16>
}

// -----

#gpu_lds_addrspace = #gpu.address_space<workgroup>
#gpu_global_addrspace = #gpu.address_space<global>

// CHECK: #[[BOTH_MAP:.*]] = affine_map<()[s0] -> (s0 + 32)>
// CHECK: #[[BOTH_MAP1:.*]] = affine_map<()[s0] -> (s0 + 64)>

// CHECK: func @test_make_dma_base_both_fold
// CHECK-SAME: %[[MEM:.*]]: memref<64x128xf16, #gpu.address_space<global>>, %[[LDS:.*]]: memref<4096xf16, #gpu.address_space<workgroup>>, %[[GLOBAL_I:.*]]: index, %[[GLOBAL_J:.*]]: index, %[[LDS_I:.*]]: index, %[[LDS_J:.*]]: index
func.func @test_make_dma_base_both_fold(%mem: memref<64x128xf16, #gpu_global_addrspace>, %lds: memref<4096xf16, #gpu_lds_addrspace>, %global_i: index, %global_j: index, %lds_i: index, %lds_j: index) -> !amdgpu.tdm_base<f16> {
  // CHECK: %[[GI:.*]] = affine.apply #[[BOTH_MAP]]()[%[[GLOBAL_I]]]
  // CHECK: %[[GJ:.*]] = affine.apply #[[BOTH_MAP1]]()[%[[GLOBAL_J]]]
  // CHECK: %[[IDX:.*]] = affine.linearize_index disjoint [%[[LDS_I]], %[[LDS_J]]] by (64, 64) : index
  // CHECK: %[[BASE:.*]] = amdgpu.make_dma_base %[[MEM]][%[[GI]], %[[GJ]]], %[[LDS]][%[[IDX]]]
  // CHECK-SAME: memref<64x128xf16, #gpu.address_space<global>>, memref<4096xf16, #gpu.address_space<workgroup>> -> !amdgpu.tdm_base<f16>
  // CHECK: return %[[BASE]] : !amdgpu.tdm_base<f16>

  %subview = memref.subview %mem[32, 64][32, 64][1, 1] : memref<64x128xf16, #gpu_global_addrspace> to memref<32x64xf16, strided<[128, 1], offset: 4160>, #gpu_global_addrspace>
  %expand_lds = memref.expand_shape %lds [[0, 1]] output_shape [64, 64] : memref<4096xf16, #gpu_lds_addrspace> into memref<64x64xf16, #gpu_lds_addrspace>
  %base = amdgpu.make_dma_base %subview[%global_i, %global_j], %expand_lds[%lds_i, %lds_j]
    : memref<32x64xf16, strided<[128, 1], offset: 4160>, #gpu_global_addrspace>, memref<64x64xf16, #gpu_lds_addrspace> -> !amdgpu.tdm_base<f16>
  func.return %base : !amdgpu.tdm_base<f16>
}

// -----

#gpu_lds_addrspace = #gpu.address_space<workgroup>
#gpu_global_addrspace = #gpu.address_space<global>

// CHECK: func @test_make_dma_base_nop
// CHECK-SAME: %[[MEM:.*]]: memref<64x128xf16, #gpu.address_space<global>>, %[[LDS:.*]]: memref<64x64xf16, #gpu.address_space<workgroup>>, %[[GLOBAL_I:.*]]: index, %[[GLOBAL_J:.*]]: index, %[[LDS_I:.*]]: index, %[[LDS_J:.*]]: index
func.func @test_make_dma_base_nop(%mem: memref<64x128xf16, #gpu_global_addrspace>, %lds: memref<64x64xf16, #gpu_lds_addrspace>, %global_i: index, %global_j: index, %lds_i: index, %lds_j: index) -> !amdgpu.tdm_base<f16> {
  // CHECK: %[[BASE:.*]] = amdgpu.make_dma_base %[[MEM]][%[[GLOBAL_I]], %[[GLOBAL_J]]], %[[LDS]][%[[LDS_I]], %[[LDS_J]]]
  // CHECK-SAME: memref<64x128xf16, #gpu.address_space<global>>, memref<64x64xf16, #gpu.address_space<workgroup>> -> !amdgpu.tdm_base<f16>
  // CHECK: return %[[BASE]] : !amdgpu.tdm_base<f16>
  // CHECK-NOT: subview
  // CHECK-NOT: expand_shape
  // CHECK-NOT: collapse_shape

  %base = amdgpu.make_dma_base %mem[%global_i, %global_j], %lds[%lds_i, %lds_j]
    : memref<64x128xf16, #gpu_global_addrspace>, memref<64x64xf16, #gpu_lds_addrspace> -> !amdgpu.tdm_base<f16>
  func.return %base : !amdgpu.tdm_base<f16>
}

// -----

#gpu_lds_addrspace = #gpu.address_space<workgroup>
#gpu_global_addrspace = #gpu.address_space<global>

// CHECK: func @test_make_gather_dma_base_nop
// CHECK-SAME: %[[MEM:.*]]: memref<64x128xf16, #gpu.address_space<global>>, %[[LDS:.*]]: memref<64x64xf16, #gpu.address_space<workgroup>>, %[[GLOBAL_I:.*]]: index, %[[GLOBAL_J:.*]]: index, %[[LDS_I:.*]]: index, %[[LDS_J:.*]]: index
func.func @test_make_gather_dma_base_nop(%mem: memref<64x128xf16, #gpu_global_addrspace>, %lds: memref<64x64xf16, #gpu_lds_addrspace>, %global_i: index, %global_j: index, %lds_i: index, %lds_j: index) -> !amdgpu.tdm_gather_base<f16, i16> {
  // CHECK: %[[BASE:.*]] = amdgpu.make_gather_dma_base %[[MEM]][%[[GLOBAL_I]], %[[GLOBAL_J]]], %[[LDS]][%[[LDS_I]], %[[LDS_J]]]
  // CHECK-SAME: memref<64x128xf16, #gpu.address_space<global>>, memref<64x64xf16, #gpu.address_space<workgroup>> -> !amdgpu.tdm_gather_base<f16, i16>
  // CHECK: return %[[BASE]] : !amdgpu.tdm_gather_base<f16, i16>
  // CHECK-NOT: subview
  // CHECK-NOT: expand_shape
  // CHECK-NOT: collapse_shape

  %base = amdgpu.make_gather_dma_base %mem[%global_i, %global_j], %lds[%lds_i, %lds_j]
    : memref<64x128xf16, #gpu_global_addrspace>, memref<64x64xf16, #gpu_lds_addrspace> -> !amdgpu.tdm_gather_base<f16, i16>
  func.return %base : !amdgpu.tdm_gather_base<f16, i16>
}

// -----

#gpu_lds_addrspace = #gpu.address_space<workgroup>
#gpu_global_addrspace = #gpu.address_space<global>

// CHECK: #[[GLOBAL_ASYNC_MAP:.*]] = affine_map<()[s0] -> (s0 + 32)>
// CHECK: #[[GLOBAL_ASYNC_MAP1:.*]] = affine_map<()[s0] -> (s0 + 64)>

// CHECK: func @test_global_load_async_to_lds_both_fold_masked
// CHECK-SAME: %[[SRC:.*]]: memref<64x128xf32, #gpu.address_space<global>>, %[[LDS:.*]]: memref<4096xf32, #gpu.address_space<workgroup>>, %[[GLOBAL_I:.*]]: index, %[[GLOBAL_J:.*]]: index, %[[LDS_I:.*]]: index, %[[LDS_J:.*]]: index, %[[MASK:.*]]: i1
func.func @test_global_load_async_to_lds_both_fold_masked(%src: memref<64x128xf32, #gpu_global_addrspace>, %lds: memref<4096xf32, #gpu_lds_addrspace>, %global_i: index, %global_j: index, %lds_i: index, %lds_j: index, %mask: i1) {
  // CHECK: %[[GI:.*]] = affine.apply #[[GLOBAL_ASYNC_MAP]]()[%[[GLOBAL_I]]]
  // CHECK: %[[GJ:.*]] = affine.apply #[[GLOBAL_ASYNC_MAP1]]()[%[[GLOBAL_J]]]
  // CHECK: %[[LDS_IDX:.*]] = affine.linearize_index [%[[LDS_I]], %[[LDS_J]]] by (64, 64) : index
  // CHECK: amdgpu.global_load_async_to_lds %[[SRC]][%[[GI]], %[[GJ]]], %[[LDS]][%[[LDS_IDX]]], %[[MASK]]
  // CHECK-SAME: vector<4xf32>, memref<64x128xf32, #gpu.address_space<global>>, memref<4096xf32, #gpu.address_space<workgroup>>

  %subview = memref.subview %src[32, 64][32, 64][1, 1]
    : memref<64x128xf32, #gpu_global_addrspace>
    to memref<32x64xf32, strided<[128, 1], offset: 4160>, #gpu_global_addrspace>
  %expand_lds = memref.expand_shape %lds [[0, 1]] output_shape [64, 64]
    : memref<4096xf32, #gpu_lds_addrspace>
    into memref<64x64xf32, #gpu_lds_addrspace>
  amdgpu.global_load_async_to_lds %subview[%global_i, %global_j], %expand_lds[%lds_i, %lds_j], %mask
    : vector<4xf32>, memref<32x64xf32, strided<[128, 1], offset: 4160>, #gpu_global_addrspace>,
      memref<64x64xf32, #gpu_lds_addrspace>
  func.return
}

// -----

#gpu_lds_addrspace = #gpu.address_space<workgroup>
#gpu_global_addrspace = #gpu.address_space<global>

// CHECK: func @test_global_load_async_to_lds_no_mask_dst_collapse
// CHECK-SAME: %[[SRC:.*]]: memref<8192xi32, #gpu.address_space<global>>, %[[LDS:.*]]: memref<64x64xi32, #gpu.address_space<workgroup>>, %[[SRC_IDX:.*]]: index, %[[DST_IDX:.*]]: index
func.func @test_global_load_async_to_lds_no_mask_dst_collapse(%src: memref<8192xi32, #gpu_global_addrspace>, %lds: memref<64x64xi32, #gpu_lds_addrspace>, %src_idx: index, %dst_idx: index) {
  // CHECK: %[[INDICES:.*]]:2 = affine.delinearize_index %[[DST_IDX]] into (64, 64) : index, index
  // CHECK: amdgpu.global_load_async_to_lds %[[SRC]][%[[SRC_IDX]]], %[[LDS]][%[[INDICES]]#0, %[[INDICES]]#1] :
  // CHECK-SAME: i32, memref<8192xi32, #gpu.address_space<global>>, memref<64x64xi32, #gpu.address_space<workgroup>>

  %collapse_lds = memref.collapse_shape %lds [[0, 1]]
    : memref<64x64xi32, #gpu_lds_addrspace>
    into memref<4096xi32, #gpu_lds_addrspace>
  amdgpu.global_load_async_to_lds %src[%src_idx], %collapse_lds[%dst_idx]
    : i32, memref<8192xi32, #gpu_global_addrspace>,
      memref<4096xi32, #gpu_lds_addrspace>
  func.return
}

// -----

#gpu_lds_addrspace = #gpu.address_space<workgroup>
#gpu_global_addrspace = #gpu.address_space<global>

// CHECK: func @test_global_load_async_to_lds_masked_dst_collapse
// CHECK-SAME: %[[SRC:.*]]: memref<8192xi32, #gpu.address_space<global>>, %[[LDS:.*]]: memref<64x64xi32, #gpu.address_space<workgroup>>, %[[SRC_IDX:.*]]: index, %[[DST_IDX:.*]]: index, %[[MASK:.*]]: i1
func.func @test_global_load_async_to_lds_masked_dst_collapse(%src: memref<8192xi32, #gpu_global_addrspace>, %lds: memref<64x64xi32, #gpu_lds_addrspace>, %src_idx: index, %dst_idx: index, %mask: i1) {
  // CHECK: %[[INDICES:.*]]:2 = affine.delinearize_index %[[DST_IDX]] into (64) : index, index
  // CHECK: amdgpu.global_load_async_to_lds %[[SRC]][%[[SRC_IDX]]], %[[LDS]][%[[INDICES]]#0, %[[INDICES]]#1], %[[MASK]] :
  // CHECK-SAME: i32, memref<8192xi32, #gpu.address_space<global>>, memref<64x64xi32, #gpu.address_space<workgroup>>

  %collapse_lds = memref.collapse_shape %lds [[0, 1]]
    : memref<64x64xi32, #gpu_lds_addrspace>
    into memref<4096xi32, #gpu_lds_addrspace>
  amdgpu.global_load_async_to_lds %src[%src_idx], %collapse_lds[%dst_idx], %mask
    : i32, memref<8192xi32, #gpu_global_addrspace>,
      memref<4096xi32, #gpu_lds_addrspace>
  func.return
}

// -----

#gpu_global_addrspace = #gpu.address_space<global>

// CHECK: func @test_global_transpose_load_expand_shape
// CHECK-SAME: %[[SRC:.*]]: memref<8192xf16, #gpu.address_space<global>>, %[[I:.*]]: index, %[[J:.*]]: index
func.func @test_global_transpose_load_expand_shape(%src: memref<8192xf16, #gpu_global_addrspace>, %i: index, %j: index) -> vector<8xf16> {
  // CHECK: %[[IDX:.*]] = affine.linearize_index disjoint [%[[I]], %[[J]]] by (64, 128) : index
  // CHECK: amdgpu.global_transpose_load %[[SRC]][%[[IDX]]]
  // CHECK-SAME: memref<8192xf16, #gpu.address_space<global>> -> vector<8xf16>

  %expand_src = memref.expand_shape %src [[0, 1]] output_shape [64, 128]
    : memref<8192xf16, #gpu_global_addrspace>
    into memref<64x128xf16, #gpu_global_addrspace>
  %result = amdgpu.global_transpose_load %expand_src[%i, %j]
    : memref<64x128xf16, #gpu_global_addrspace> -> vector<8xf16>
  return %result : vector<8xf16>
}

// -----

#gpu_wg = #gpu.address_space<workgroup>

// CHECK: func @test_ds_barrier_ops_expand_shape
// CHECK-SAME: %[[BARRIER:.*]]: memref<16x!amdgpu.ds_barrier_state, #gpu.address_space<workgroup>>, %[[I:.*]]: index, %[[J:.*]]: index, %[[PARTICIPANTS:.*]]: i32, %[[COUNT:.*]]: i64
func.func @test_ds_barrier_ops_expand_shape(%barrier: memref<16x!amdgpu.ds_barrier_state, #gpu_wg>, %i: index, %j: index, %participants: i32, %count: i64) -> !amdgpu.ds_barrier_state {
  // CHECK: %[[INIT_IDX:.*]] = affine.linearize_index disjoint [%[[I]], %[[J]]] by (4, 4) : index
  // CHECK: amdgpu.ds_barrier_init %[[BARRIER]][%[[INIT_IDX]]], %[[PARTICIPANTS]]
  // CHECK-SAME: memref<16x!amdgpu.ds_barrier_state, #gpu.address_space<workgroup>>, i32
  // CHECK: %[[POLL_IDX:.*]] = affine.linearize_index disjoint [%[[I]], %[[J]]] by (4, 4) : index
  // CHECK: %[[STATE:.*]] = amdgpu.ds_barrier_poll_state %[[BARRIER]][%[[POLL_IDX]]]
  // CHECK-SAME: memref<16x!amdgpu.ds_barrier_state, #gpu.address_space<workgroup>> -> !amdgpu.ds_barrier_state
  // CHECK: %[[ASYNC_IDX:.*]] = affine.linearize_index disjoint [%[[I]], %[[J]]] by (4, 4) : index
  // CHECK: amdgpu.ds_async_barrier_arrive %[[BARRIER]][%[[ASYNC_IDX]]]
  // CHECK-SAME: memref<16x!amdgpu.ds_barrier_state, #gpu.address_space<workgroup>>
  // CHECK: %[[ARRIVE_IDX:.*]] = affine.linearize_index disjoint [%[[I]], %[[J]]] by (4, 4) : index
  // CHECK: %[[OLD:.*]] = amdgpu.ds_barrier_arrive %[[BARRIER]][%[[ARRIVE_IDX]]], %[[COUNT]]
  // CHECK-SAME: memref<16x!amdgpu.ds_barrier_state, #gpu.address_space<workgroup>>, i64 -> !amdgpu.ds_barrier_state

  %expand_barrier = memref.expand_shape %barrier [[0, 1]] output_shape [4, 4]
    : memref<16x!amdgpu.ds_barrier_state, #gpu_wg>
    into memref<4x4x!amdgpu.ds_barrier_state, #gpu_wg>
  amdgpu.ds_barrier_init %expand_barrier[%i, %j], %participants
    : memref<4x4x!amdgpu.ds_barrier_state, #gpu_wg>, i32
  %state = amdgpu.ds_barrier_poll_state %expand_barrier[%i, %j]
    : memref<4x4x!amdgpu.ds_barrier_state, #gpu_wg> -> !amdgpu.ds_barrier_state
  amdgpu.ds_async_barrier_arrive %expand_barrier[%i, %j]
    : memref<4x4x!amdgpu.ds_barrier_state, #gpu_wg>
  %old_state = amdgpu.ds_barrier_arrive %expand_barrier[%i, %j], %count
    : memref<4x4x!amdgpu.ds_barrier_state, #gpu_wg>, i64 -> !amdgpu.ds_barrier_state
  return %old_state : !amdgpu.ds_barrier_state
}

// -----

#gpu_wg = #gpu.address_space<workgroup>

// CHECK: func @test_make_dma_descriptor_atomic_barrier_expand_shape
// CHECK-SAME: %[[BASE:.*]]: !amdgpu.tdm_base<i32>, %[[BARRIER:.*]]: memref<16x!amdgpu.ds_barrier_state, #gpu.address_space<workgroup>>, %[[I:.*]]: index, %[[J:.*]]: index
func.func @test_make_dma_descriptor_atomic_barrier_expand_shape(%base: !amdgpu.tdm_base<i32>, %barrier: memref<16x!amdgpu.ds_barrier_state, #gpu_wg>, %i: index, %j: index) -> !amdgpu.tdm_descriptor {
  // CHECK: %[[BARRIER_IDX:.*]] = affine.linearize_index disjoint [%[[I]], %[[J]]] by (4, 4) : index
  // CHECK: %[[DESC:.*]] = amdgpu.make_dma_descriptor %[[BASE]]
  // CHECK-SAME: globalSize [64, 64]
  // CHECK-SAME: globalStride [64, 1]
  // CHECK-SAME: sharedSize [64, 64]
  // CHECK-SAME: atomicBarrier(%[[BARRIER]][%[[BARRIER_IDX]]] : memref<16x!amdgpu.ds_barrier_state, #gpu.address_space<workgroup>>)

  %expand_barrier = memref.expand_shape %barrier [[0, 1]] output_shape [4, 4]
    : memref<16x!amdgpu.ds_barrier_state, #gpu_wg>
    into memref<4x4x!amdgpu.ds_barrier_state, #gpu_wg>
  %desc = amdgpu.make_dma_descriptor %base
    globalSize [64, 64]
    globalStride [64, 1]
    sharedSize [64, 64]
    atomicBarrier(%expand_barrier[%i, %j] : memref<4x4x!amdgpu.ds_barrier_state, #gpu_wg>)
    : !amdgpu.tdm_base<i32> -> !amdgpu.tdm_descriptor
  return %desc : !amdgpu.tdm_descriptor
}

// -----

#gpu_wg = #gpu.address_space<workgroup>

// CHECK: func @test_make_gather_dma_descriptor_atomic_barrier_expand_shape
// CHECK-SAME: %[[BASE:.*]]: !amdgpu.tdm_gather_base<i32, i32>, %[[INDICES:.*]]: vector<8xi32>, %[[BARRIER:.*]]: memref<16x!amdgpu.ds_barrier_state, #gpu.address_space<workgroup>>, %[[I:.*]]: index, %[[J:.*]]: index
func.func @test_make_gather_dma_descriptor_atomic_barrier_expand_shape(%base: !amdgpu.tdm_gather_base<i32, i32>, %indices: vector<8xi32>, %barrier: memref<16x!amdgpu.ds_barrier_state, #gpu_wg>, %i: index, %j: index) -> !amdgpu.tdm_descriptor {
  // CHECK: %[[BARRIER_IDX:.*]] = affine.linearize_index disjoint [%[[I]], %[[J]]] by (4, 4) : index
  // CHECK: %[[DESC:.*]] = amdgpu.make_gather_dma_descriptor %[[BASE]][%[[INDICES]]]
  // CHECK-SAME: globalSize [64, 64]
  // CHECK-SAME: globalStride [64, 1]
  // CHECK-SAME: sharedSize [64, 64]
  // CHECK-SAME: atomicBarrier(%[[BARRIER]][%[[BARRIER_IDX]]] : memref<16x!amdgpu.ds_barrier_state, #gpu.address_space<workgroup>>)

  %expand_barrier = memref.expand_shape %barrier [[0, 1]] output_shape [4, 4]
    : memref<16x!amdgpu.ds_barrier_state, #gpu_wg>
    into memref<4x4x!amdgpu.ds_barrier_state, #gpu_wg>
  %desc = amdgpu.make_gather_dma_descriptor %base[%indices]
    globalSize [64, 64]
    globalStride [64, 1]
    sharedSize [64, 64]
    atomicBarrier(%expand_barrier[%i, %j] : memref<4x4x!amdgpu.ds_barrier_state, #gpu_wg>)
    : !amdgpu.tdm_gather_base<i32, i32>, vector<8xi32> -> !amdgpu.tdm_descriptor
  return %desc : !amdgpu.tdm_descriptor
}

// -----

// CHECK: func @test_make_dma_descriptor_no_atomic_barrier
// CHECK-SAME: %[[BASE:.*]]: !amdgpu.tdm_base<i32>
func.func @test_make_dma_descriptor_no_atomic_barrier(%base: !amdgpu.tdm_base<i32>) -> !amdgpu.tdm_descriptor {
  // CHECK: %[[DESC:.*]] = amdgpu.make_dma_descriptor %[[BASE]]
  // CHECK-SAME: globalSize [64, 64]
  // CHECK-SAME: globalStride [64, 1]
  // CHECK-SAME: sharedSize [64, 64]
  // CHECK-NOT: atomicBarrier

  %desc = amdgpu.make_dma_descriptor %base
    globalSize [64, 64]
    globalStride [64, 1]
    sharedSize [64, 64]
    : !amdgpu.tdm_base<i32> -> !amdgpu.tdm_descriptor
  return %desc : !amdgpu.tdm_descriptor
}

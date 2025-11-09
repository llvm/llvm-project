// RUN: mlir-opt --amdgpu-fold-memrefs-ops --split-input-file %s | FileCheck %s

#gpu_lds_addrspace = 3

// CHECK: func @test_subview_folding
// CHECK-SAME: %[[ARG0:.*]]: index, %[[ARG1:.*]]: index
func.func @test_subview_folding(%offset_i: index, %offset_j: index) {
  // CHECK: %[[LOCAL:.*]] = memref.alloc() : memref<64x64xf16, 3>
  // CHECK: %[[MEM:.*]] = memref.alloc() : memref<64x128xf16>
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: amdgpu.gather_to_lds %[[MEM]][%[[ARG0]], %[[ARG1]]], %[[LOCAL]][%[[C0]], %[[C0]]]
  // CHECK-SAME: vector<8xf16>, memref<64x128xf16>, memref<64x64xf16, 3>

  %alloc = memref.alloc() : memref<64x64xf16, #gpu_lds_addrspace>
  %mem = memref.alloc() : memref<64x128xf16>
  %subview = memref.subview %mem[0, 0][32, 64][1, 1] : memref<64x128xf16> to memref<32x64xf16, strided<[128, 1]>>
  %c0 = arith.constant 0 : index
  amdgpu.gather_to_lds %subview[%offset_i, %offset_j], %alloc[%c0, %c0]
    : vector<8xf16>, memref<32x64xf16, strided<[128, 1]>>, memref<64x64xf16, #gpu_lds_addrspace>
  func.return
}

// -----

#gpu_lds_addrspace = 3

// CHECK: #[[MAP:.*]] = affine_map<()[s0] -> (s0 + 32)>
// CHECK: #[[MAP1:.*]] = affine_map<()[s0] -> (s0 + 64)>

// CHECK: func @subview_folding_offset
// CHECK-SAME: %[[ARG0:.*]]: index, %[[ARG1:.*]]: index
func.func @subview_folding_offset(%offset_i: index, %offset_j: index) {
  // CHECK: %[[LOCAL:.*]] = memref.alloc() : memref<64x64xf16, 3>
  // CHECK: %[[MEM:.*]] = memref.alloc() : memref<64x128xf16>
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[IDX0:.*]] = affine.apply #[[MAP]]()[%[[ARG0]]]
  // CHECK: %[[IDX1:.*]] = affine.apply #[[MAP1]]()[%[[ARG1]]]
  // CHECK: amdgpu.gather_to_lds %[[MEM]][%[[IDX0]], %[[IDX1]]], %[[LOCAL]][%[[C0]], %[[C0]]]
  // CHECK-SAME: vector<8xf16>, memref<64x128xf16>, memref<64x64xf16, 3>

  %alloc = memref.alloc() : memref<64x64xf16, #gpu_lds_addrspace>
  %mem = memref.alloc() : memref<64x128xf16>
  %subview = memref.subview %mem[32, 64][32, 64][1, 1] : memref<64x128xf16> to memref<32x64xf16, strided<[128, 1], offset: 4160>>
  %c0 = arith.constant 0 : index
  amdgpu.gather_to_lds %subview[%offset_i, %offset_j], %alloc[%c0, %c0]
    : vector<8xf16>, memref<32x64xf16, strided<[128, 1], offset: 4160>>, memref<64x64xf16, #gpu_lds_addrspace>
  func.return
}

// -----

#gpu_lds_addrspace = 3

// CHECK: func @test_expand_shape
// CHECK-SAME: %[[ARG0:.*]]: index, %[[ARG1:.*]]: index
func.func @test_expand_shape(%offset_i: index, %offset_j: index) {
  // CHECK: %[[LOCAL:.*]] = memref.alloc() : memref<4096xf16, 3>
  // CHECK: %[[MEM:.*]] = memref.alloc() : memref<8192xf16>
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[IDXM:.*]] = affine.linearize_index [%[[ARG0]], %[[ARG1]]] by (64, 128) : index
  // CHECK: %[[IDXL:.*]] = affine.linearize_index [%[[C0]], %[[C0]]] by (64, 64) : index
  // CHECK: amdgpu.gather_to_lds %[[MEM]][%[[IDXM]]], %[[LOCAL]][%[[IDXL]]]
  // CHECK-SAME: vector<8xf16>, memref<8192xf16>, memref<4096xf16, 3>

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

#gpu_lds_addrspace = 3

// CHECK: func @test_collapse_shape
// CHECK-SAME: %[[ARG0:.*]]: index, %[[ARG1:.*]]: index
func.func @test_collapse_shape(%offset_i: index, %offset_j: index) {
  // CHECK: %[[LOCAL:.*]] = memref.alloc() : memref<64x64xf16, 3>
  // CHECK: %[[MEM:.*]] = memref.alloc() : memref<64x128xf16>
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[INDICES_MEM:.*]]:2 = affine.delinearize_index %[[ARG0]] into (64, 128) : index, index
  // CHECK: %[[INDICES_LDS:.*]]:2 = affine.delinearize_index %[[ARG1]] into (64, 64) : index, index
  // CHECK: amdgpu.gather_to_lds %[[MEM]][%[[INDICES_MEM]]#0, %[[INDICES_MEM]]#1], %[[LOCAL]][%[[INDICES_LDS]]#0, %[[INDICES_LDS]]#1]
  // CHECK-SAME: vector<8xf16>, memref<64x128xf16>, memref<64x64xf16, 3>

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

#gpu_lds_addrspace = 3


// CHECK: func @test_expand_shape_src_raw_buffer
// CHECK-SAME: %[[ARG0:.*]]: memref<8192xf16, #amdgpu.address_space<fat_raw_buffer>>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index
func.func @test_expand_shape_src_raw_buffer(%mem : memref<8192xf16, #amdgpu.address_space<fat_raw_buffer>>, %offset_i: index, %offset_j: index) {
  // CHECK: %[[LOCAL:.*]] = memref.alloc() : memref<4096xf16, 3>
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[IDXM:.*]] = affine.linearize_index [%[[ARG1]], %[[ARG2]]] by (64, 128) : index
  // CHECK: amdgpu.gather_to_lds %[[ARG0]][%[[IDXM]]], %[[LOCAL]][%[[C0]]]
  // CHECK-SAME: vector<8xf16>, memref<8192xf16, #amdgpu.address_space<fat_raw_buffer>>, memref<4096xf16, 3>

  %alloc = memref.alloc() : memref<4096xf16, #gpu_lds_addrspace>
  %expand_mem = memref.expand_shape %mem [[0, 1]] output_shape [64, 128] : memref<8192xf16, #amdgpu.address_space<fat_raw_buffer>> into memref<64x128xf16, #amdgpu.address_space<fat_raw_buffer>>

  %c0 = arith.constant 0 : index
  amdgpu.gather_to_lds %expand_mem[%offset_i, %offset_j], %alloc[%c0]
    : vector<8xf16>, memref<64x128xf16, #amdgpu.address_space<fat_raw_buffer>>, memref<4096xf16, #gpu_lds_addrspace>
  func.return
}

// -----

#gpu_lds_addrspace = 3

// CHECK: func @test_expand_shape_dst_only
// CHECK-SAME: %[[ARG0:.*]]: index, %[[ARG1:.*]]: index
func.func @test_expand_shape_dst_only(%offset_i: index, %offset_j: index) {
  // CHECK: %[[LOCAL:.*]] = memref.alloc() : memref<4096xf16, 3>
  // CHECK: %[[MEM:.*]] = memref.alloc() : memref<8192xf16>
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[IDX_LDS:.*]] = affine.linearize_index [%[[ARG1]], %[[C0]]] by (64, 64) : index
  // CHECK: amdgpu.gather_to_lds %[[MEM]][%[[ARG0]]], %[[LOCAL]][%[[IDX_LDS]]]
  // CHECK-SAME: vector<8xf16>, memref<8192xf16>, memref<4096xf16, 3>

  %alloc = memref.alloc() : memref<4096xf16, #gpu_lds_addrspace>
  %mem = memref.alloc() : memref<8192xf16>
  %expand_alloc = memref.expand_shape %alloc [[0, 1]] output_shape [64, 64] : memref<4096xf16, #gpu_lds_addrspace> into memref<64x64xf16, #gpu_lds_addrspace>

  %c0 = arith.constant 0 : index
  amdgpu.gather_to_lds %mem[%offset_i], %expand_alloc[%offset_j, %c0]
    : vector<8xf16>, memref<8192xf16>, memref<64x64xf16, #gpu_lds_addrspace>
  func.return
}

// -----

#gpu_lds_addrspace = 3

// CHECK: func @test_nop
// CHECK-SAME: %[[ARG0:.*]]: memref<8192xf16, #amdgpu.address_space<fat_raw_buffer>>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index
func.func @test_nop(%mem : memref<8192xf16, #amdgpu.address_space<fat_raw_buffer>>, %offset_i: index, %offset_j: index) {
  // CHECK: %[[LOCAL:.*]] = memref.alloc() : memref<4096xf16, 3>
  // CHECK: amdgpu.gather_to_lds %[[ARG0]][%[[ARG1]]], %[[LOCAL]][%[[ARG2]]]
  // CHECK-SAME: vector<8xf16>, memref<8192xf16, #amdgpu.address_space<fat_raw_buffer>>, memref<4096xf16, 3>

  %alloc = memref.alloc() : memref<4096xf16, #gpu_lds_addrspace>
  amdgpu.gather_to_lds %mem[%offset_i], %alloc[%offset_j]
    : vector<8xf16>, memref<8192xf16, #amdgpu.address_space<fat_raw_buffer>>, memref<4096xf16, #gpu_lds_addrspace>
  func.return
}

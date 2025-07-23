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
  // CHECK: %[[LOCAL:.*]] = memref.alloc() : memref<64x64xf16, 3>
  // CHECK: %[[MEM:.*]] = memref.alloc() : memref<8192xf16>
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[IDX:.*]] = affine.linearize_index [%[[ARG0]], %[[ARG1]]] by (64, 128) : index
  // CHECK: amdgpu.gather_to_lds %[[MEM]][%[[IDX]]], %[[LOCAL]][%[[C0]], %[[C0]]]
  // CHECK-SAME: vector<8xf16>, memref<8192xf16>, memref<64x64xf16, 3>

  %alloc = memref.alloc() : memref<64x64xf16, #gpu_lds_addrspace>
  %mem = memref.alloc() : memref<8192xf16>
  %expand = memref.expand_shape %mem [[0, 1]] output_shape [64, 128] : memref<8192xf16> into memref<64x128xf16>
  %c0 = arith.constant 0 : index
  amdgpu.gather_to_lds %expand[%offset_i, %offset_j], %alloc[%c0, %c0]
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
  // CHECK: %[[INDICES:.*]]:2 = affine.delinearize_index %[[ARG0]] into (64, 128) : index, index
  // CHECK: amdgpu.gather_to_lds %[[MEM]][%[[INDICES]]#0, %[[INDICES]]#1], %[[LOCAL]][%[[C0]], %[[C0]]]
  // CHECK-SAME: vector<8xf16>, memref<64x128xf16>, memref<64x64xf16, 3>

  %alloc = memref.alloc() : memref<64x64xf16, #gpu_lds_addrspace>
  %mem = memref.alloc() : memref<64x128xf16>
  %collapse = memref.collapse_shape %mem [[0, 1]] : memref<64x128xf16> into memref<8192xf16>
  %c0 = arith.constant 0 : index
  amdgpu.gather_to_lds %collapse[%offset_i], %alloc[%c0, %c0]
    : vector<8xf16>, memref<8192xf16>, memref<64x64xf16, #gpu_lds_addrspace>
  func.return
}

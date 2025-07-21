// RUN: mlir-opt -amdgpu-fold-subview-ops -split-input-file %s | FileCheck %s

#gpu_lds_addrspace = 3

// CHECK: func @test_memref
// CHECK-SAME: %[[ARG0:.*]]: index, %[[ARG1:.*]]: index
func.func @test_memref(%offset_i: index, %offset_j: index) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[LOCAL:.*]] = memref.alloc() : memref<64x64xf16, 3>
  // CHECK: %[[MEM:.*]] = memref.alloc() : memref<64x128xf16>
  // CHECK:  amdgpu.gather_to_lds %[[MEM]][%arg0, %arg1], %[[LOCAL]][%[[C0]], %[[C0]]]
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
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[LOCAL:.*]] = memref.alloc() : memref<64x64xf16, 3>
  // CHECK: %[[MEM:.*]] = memref.alloc() : memref<64x128xf16>

  // CHECK: %[[IDX0:.*]] = affine.apply #[[MAP]]()[%[[ARG0]]]
  // CHECK: %[[IDX1:.*]] = affine.apply #[[MAP1]]()[%[[ARG1]]]

  // CHECK:  amdgpu.gather_to_lds %[[MEM]][%[[IDX0]], %[[IDX1]]], %[[LOCAL]][%[[C0]], %[[C0]]]
  // CHECK-SAME: vector<8xf16>, memref<64x128xf16>, memref<64x64xf16, 3>

  %alloc = memref.alloc() : memref<64x64xf16, #gpu_lds_addrspace>
  %mem = memref.alloc() : memref<64x128xf16>
  %subview = memref.subview %mem[32, 64][32, 64][1, 1] : memref<64x128xf16> to memref<32x64xf16, strided<[128, 1], offset: 4160>>
  %c0 = arith.constant 0 : index
  amdgpu.gather_to_lds %subview[%offset_i, %offset_j], %alloc[%c0, %c0]
    : vector<8xf16>, memref<32x64xf16, strided<[128, 1], offset: 4160>>, memref<64x64xf16, #gpu_lds_addrspace>
  func.return
}

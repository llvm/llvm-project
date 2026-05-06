// RUN: mlir-opt --canonicalize --split-input-file %s | FileCheck %s

// CHECK-LABEL: @make_dma_descriptor_fold
// CHECK-SAME: (%[[BASE:.+]]: !amdgpu.tdm_base<i32>, %[[IDX:.+]]: index, %[[I32:.+]]: i32)
func.func @make_dma_descriptor_fold(%base: !amdgpu.tdm_base<i32>, %idx: index, %i32: i32) -> !amdgpu.tdm_descriptor {
  %c64 = arith.constant 64 : index

  // CHECK: amdgpu.make_dma_descriptor %[[BASE]]
  %0 = amdgpu.make_dma_descriptor %base
        // CHECK-SAME: globalSize [64, 64]
        globalSize [%c64, %c64]
        // CHECK-SAME: globalStride [64, 1]
        globalStride [%c64, 1]
        // CHECK-SAME: sharedSize [64, 64]
        sharedSize [%c64, %c64]
        iterate %idx, %i32, %idx
        : !amdgpu.tdm_base<i32> -> !amdgpu.tdm_descriptor
  func.return %0 : !amdgpu.tdm_descriptor
}

// -----

// CHECK-LABEL: @make_gather_dma_descriptor_fold
// CHECK-SAME: (%[[BASE:.+]]: !amdgpu.tdm_gather_base<i32, i32>, %[[IDX:.+]]: index, %[[I32:.+]]: i32, %[[INDICES:.+]]: vector<8xi32>)
func.func @make_gather_dma_descriptor_fold(%base: !amdgpu.tdm_gather_base<i32, i32>, %idx: index, %i32: i32, %indices: vector<8xi32>) -> !amdgpu.tdm_descriptor {
  %c64 = arith.constant 64 : index

  // CHECK: amdgpu.make_gather_dma_descriptor %[[BASE]][%[[INDICES]]]
  %0 = amdgpu.make_gather_dma_descriptor %base [%indices]
        // CHECK-SAME: globalSize [64, 64]
        globalSize [%c64, %c64]
        // CHECK-SAME: globalStride [64, 1]
        globalStride [%c64, 1]
        // CHECK-SAME: sharedSize [64, 64]
        sharedSize [%c64, %c64]
        iterate %idx, %i32, %idx
        : !amdgpu.tdm_gather_base<i32, i32>, vector<8xi32> -> !amdgpu.tdm_descriptor
  func.return %0 : !amdgpu.tdm_descriptor
}

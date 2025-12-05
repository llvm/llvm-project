// RUN: mlir-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: @make_dma_descriptor_fold
// CHECK-SAME: (%[[BASE:.+]]: !amdgpu.tdm_base<i32>, %[[IDX:.+]]: index)
func.func @make_dma_descriptor_fold(%base: !amdgpu.tdm_base<i32>, %idx: index) -> !amdgpu.tdm_descriptor {
  %c64 = arith.constant 64 : index

  // CHECK: amdgpu.make_dma_descriptor %[[BASE]]
  %0 = amdgpu.make_dma_descriptor %base
        // CHECK-SAME: globalSize [64, 64]
        globalSize [%c64, %c64]
        // CHECK-SAME: globalStride [64, 1]
        globalStride [%c64, 1]
        // CHECK-SAME: sharedSize [64, 64]
        sharedSize [%c64, %c64]
        iterate %idx, %idx, %idx
        : !amdgpu.tdm_base<i32> -> !amdgpu.tdm_descriptor
  func.return %0 : !amdgpu.tdm_descriptor
}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_subgroups %s -o - | FileCheck %s

; CHECK-DAG: Capability SubgroupShuffleINTEL
; CHECK-DAG: Capability SubgroupBufferBlockIOINTEL
; CHECK-DAG: Capability SubgroupImageBlockIOINTEL
; CHECK: Extension "SPV_INTEL_subgroups"

; CHECK-DAG: %[[#Float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#FloatVec:]] = OpTypeVector %[[#Float]] 2
; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#IntVec:]] = OpTypeVector %[[#Int]] 2

; CHECK: Function
; CHECK: %[[#X:]] = OpFunctionParameter
; CHECK: %[[#C:]] = OpFunctionParameter
; CHECK: %[[#ImgIn:]] = OpFunctionParameter
; CHECK: %[[#ImgOut:]] = OpFunctionParameter
; CHECK: %[[#Coord:]] = OpFunctionParameter
; CHECK: %[[#Ptr:]] = OpFunctionParameter

; CHECK: %[[#]] = OpSubgroupShuffleINTEL %[[#FloatVec]] %[[#X]] %[[#C]]
; CHECK: %[[#]] = OpSubgroupShuffleDownINTEL %[[#FloatVec]] %[[#X]] %[[#X]] %[[#C]]
; CHECK: %[[#]] = OpSubgroupShuffleUpINTEL %[[#FloatVec]] %[[#X]] %[[#X]] %[[#C]]
; CHECK: %[[#]] = OpSubgroupShuffleXorINTEL %[[#FloatVec]] %[[#X]] %[[#C]]
; CHECK: %[[#ResImg1:]] = OpSubgroupImageBlockReadINTEL %[[#IntVec]] %[[#ImgIn]] %[[#Coord]]
; CHECK: OpSubgroupImageBlockWriteINTEL %[[#ImgOut]] %[[#Coord]] %[[#ResImg1]]
; CHECK: %[[#Res1:]] = OpSubgroupBlockReadINTEL %[[#IntVec]] %[[#Ptr]]
; CHECK: OpSubgroupBlockWriteINTEL %[[#Ptr]] %[[#Res1]]
; CHECK: %[[#]] = OpSubgroupShuffleINTEL %[[#FloatVec]] %[[#X]] %[[#C]]
; CHECK: %[[#]] = OpSubgroupShuffleDownINTEL %[[#FloatVec]] %[[#X]] %[[#X]] %[[#C]]
; CHECK: %[[#]] = OpSubgroupShuffleUpINTEL %[[#FloatVec]] %[[#X]] %[[#X]] %[[#C]]
; CHECK: %[[#]] = OpSubgroupShuffleXorINTEL %[[#FloatVec]] %[[#X]] %[[#C]]
; CHECK: %[[#ResImg2:]] = OpSubgroupImageBlockReadINTEL %[[#IntVec]] %[[#ImgIn]] %[[#Coord]]
; CHECK: OpSubgroupImageBlockWriteINTEL %[[#ImgOut]] %[[#Coord]] %[[#ResImg2]]
; CHECK: %[[#Res2:]] = OpSubgroupBlockReadINTEL %[[#IntVec]] %[[#Ptr]]
; CHECK: OpSubgroupBlockWriteINTEL %[[#Ptr]] %[[#Res2]]
; CHECK: Return

define spir_kernel void @test(<2 x float> %x, i32 %c, ptr addrspace(1) %image_in, ptr addrspace(1) %image_out, <2 x i32> %coord, ptr addrspace(1) %p) {
entry:
  %wrap = tail call spir_func <2 x float> @__spirv_SubgroupShuffleINTEL(<2 x float> %x, i32 %c)
  %wrap1 = tail call spir_func <2 x float> @__spirv_SubgroupShuffleDownINTEL(<2 x float> %x, <2 x float> %x, i32 %c)
  %wrap2 = tail call spir_func <2 x float> @__spirv_SubgroupShuffleUpINTEL(<2 x float> %x, <2 x float> %x, i32 %c)
  %wrap3 = tail call spir_func <2 x float> @__spirv_SubgroupShuffleXorINTEL(<2 x float> %x, i32 %c)

  %wrap4 = tail call spir_func <2 x i32> @__spirv_SubgroupImageBlockReadINTEL(ptr addrspace(1) %image_in, <2 x i32> %coord)
  tail call spir_func void @__spirv_SubgroupImageBlockWriteINTEL(ptr addrspace(1) %image_out, <2 x i32> %coord, <2 x i32> %wrap4)
  %wrap5 = tail call spir_func <2 x i32> @__spirv_SubgroupBlockReadINTEL(ptr addrspace(1) %p)
  tail call spir_func void @__spirv_SubgroupBlockWriteINTEL(ptr addrspace(1) %p, <2 x i32> %wrap5)

  %ocl = tail call spir_func <2 x float> @intel_sub_group_shuffle(<2 x float> %x, i32 %c)
  %ocl1 = tail call spir_func <2 x float> @intel_sub_group_shuffle_down(<2 x float> %x, <2 x float> %x, i32 %c)
  %ocl2 = tail call spir_func <2 x float> @intel_sub_group_shuffle_up(<2 x float> %x, <2 x float> %x, i32 %c)
  %ocl3 = tail call spir_func <2 x float> @intel_sub_group_shuffle_xor(<2 x float> %x, i32 %c)

  %ocl4 = tail call spir_func <2 x i32> @_Z27intel_sub_group_block_read214ocl_image2d_roDv2_i(ptr addrspace(1) %image_in, <2 x i32> %coord)
  tail call spir_func void @_Z28intel_sub_group_block_write214ocl_image2d_woDv2_iDv2_j(ptr addrspace(1) %image_out, <2 x i32> %coord, <2 x i32> %ocl4)
  %ocl5 = tail call spir_func <2 x i32> @intel_sub_group_block_read(ptr addrspace(1) %p)
  tail call spir_func void @intel_sub_group_block_write(ptr addrspace(1) %p, <2 x i32> %ocl5)

  ret void
}

declare spir_func <2 x float> @__spirv_SubgroupShuffleINTEL(<2 x float>, i32)
declare spir_func <2 x float> @__spirv_SubgroupShuffleDownINTEL(<2 x float>, <2 x float>, i32)
declare spir_func <2 x float> @__spirv_SubgroupShuffleUpINTEL(<2 x float>, <2 x float>, i32)
declare spir_func <2 x float> @__spirv_SubgroupShuffleXorINTEL(<2 x float>, i32)

declare spir_func <2 x i32> @__spirv_SubgroupBlockReadINTEL(ptr addrspace(1))
declare spir_func void @__spirv_SubgroupBlockWriteINTEL(ptr addrspace(1), <2 x i32>)

declare spir_func <2 x i32> @__spirv_SubgroupImageBlockReadINTEL(ptr addrspace(1), <2 x i32>)
declare spir_func void @__spirv_SubgroupImageBlockWriteINTEL(ptr addrspace(1), <2 x i32>, <2 x i32>)

declare spir_func <2 x float> @intel_sub_group_shuffle(<2 x float>, i32)
declare spir_func <2 x float> @intel_sub_group_shuffle_down(<2 x float>, <2 x float>, i32)
declare spir_func <2 x float> @intel_sub_group_shuffle_up(<2 x float>, <2 x float>, i32)
declare spir_func <2 x float> @intel_sub_group_shuffle_xor(<2 x float>, i32)

declare spir_func <2 x i32> @intel_sub_group_block_read(ptr addrspace(1))
declare spir_func void @intel_sub_group_block_write(ptr addrspace(1), <2 x i32>)

declare spir_func <2 x i32> @_Z27intel_sub_group_block_read214ocl_image2d_roDv2_i(ptr addrspace(1), <2 x i32>)
declare spir_func void @_Z28intel_sub_group_block_write214ocl_image2d_woDv2_iDv2_j(ptr addrspace(1), <2 x i32>, <2 x i32>)

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_media_block_io %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_media_block_io %s -o - -filetype=obj | spirv-val %}

; CHECK: Capability SubgroupImageMediaBlockIOINTEL
; CHECK: Extension "SPV_INTEL_media_block_io"
; CHECK-COUNT-14: SubgroupImageMediaBlockReadINTEL
; CHECK-COUNT-14: SubgroupImageMediaBlockWriteINTEL

define spir_kernel void @intel_media_block_test(<2 x i32> %edgeCoord, ptr addrspace(1) %image_in, ptr addrspace(1) %image_out) !kernel_arg_addr_space !6 !kernel_arg_access_qual !7 !kernel_arg_type !8 !kernel_arg_base_type !8 {
entry:
  %call = call spir_func i8 @_Z46__spirv_SubgroupImageMediaBlockReadINTEL_RcharPU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(ptr addrspace(1) %image_in, <2 x i32> %edgeCoord, i32 1, i32 16)
  %call1 = call spir_func <2 x i8> @_Z47__spirv_SubgroupImageMediaBlockReadINTEL_Rchar2PU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(ptr addrspace(1) %image_in, <2 x i32> %edgeCoord, i32 1, i32 16)
  %call2 = call spir_func <4 x i8> @_Z47__spirv_SubgroupImageMediaBlockReadINTEL_Rchar4PU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(ptr addrspace(1) %image_in, <2 x i32> %edgeCoord, i32 1, i32 16)
  %call3 = call spir_func <8 x i8> @_Z47__spirv_SubgroupImageMediaBlockReadINTEL_Rchar8PU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(ptr addrspace(1) %image_in, <2 x i32> %edgeCoord, i32 1, i32 16)
  %call4 = call spir_func <16 x i8> @_Z48__spirv_SubgroupImageMediaBlockReadINTEL_Rchar16PU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(ptr addrspace(1) %image_in, <2 x i32> %edgeCoord, i32 1, i32 16)
  %call5 = call spir_func i16 @_Z47__spirv_SubgroupImageMediaBlockReadINTEL_RshortPU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(ptr addrspace(1) %image_in, <2 x i32> %edgeCoord, i32 1, i32 16)
  %call6 = call spir_func <2 x i16> @_Z48__spirv_SubgroupImageMediaBlockReadINTEL_Rshort2PU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(ptr addrspace(1) %image_in, <2 x i32> %edgeCoord, i32 1, i32 16)
  %call7 = call spir_func <4 x i16> @_Z48__spirv_SubgroupImageMediaBlockReadINTEL_Rshort4PU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(ptr addrspace(1) %image_in, <2 x i32> %edgeCoord, i32 1, i32 16)
  %call8 = call spir_func <8 x i16> @_Z48__spirv_SubgroupImageMediaBlockReadINTEL_Rshort8PU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(ptr addrspace(1) %image_in, <2 x i32> %edgeCoord, i32 1, i32 16)
  %call9 = call spir_func <16 x i16> @_Z49__spirv_SubgroupImageMediaBlockReadINTEL_Rshort16PU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(ptr addrspace(1) %image_in, <2 x i32> %edgeCoord, i32 1, i32 16)
  %call10 = call spir_func i32 @_Z45__spirv_SubgroupImageMediaBlockReadINTEL_RintPU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(ptr addrspace(1) %image_in, <2 x i32> %edgeCoord, i32 1, i32 16)
  %call11 = call spir_func <2 x i32> @_Z46__spirv_SubgroupImageMediaBlockReadINTEL_Rint2PU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(ptr addrspace(1) %image_in, <2 x i32> %edgeCoord, i32 1, i32 16)
  %call12 = call spir_func <4 x i32> @_Z46__spirv_SubgroupImageMediaBlockReadINTEL_Rint4PU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(ptr addrspace(1) %image_in, <2 x i32> %edgeCoord, i32 1, i32 16)
  %call13 = call spir_func <8 x i32> @_Z46__spirv_SubgroupImageMediaBlockReadINTEL_Rint8PU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(ptr addrspace(1) %image_in, <2 x i32> %edgeCoord, i32 1, i32 16)
  call spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiic(ptr addrspace(1) %image_out, <2 x i32> %edgeCoord, i32 1, i32 16, i8 %call)
  call spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiiDv2_c(ptr addrspace(1) %image_out, <2 x i32> %edgeCoord, i32 1, i32 16, <2 x i8> %call1)
  call spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiiDv4_c(ptr addrspace(1) %image_out, <2 x i32> %edgeCoord, i32 1, i32 16, <4 x i8> %call2)
  call spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiiDv8_c(ptr addrspace(1) %image_out, <2 x i32> %edgeCoord, i32 1, i32 16, <8 x i8> %call3)
  call spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiiDv16_c(ptr addrspace(1) %image_out, <2 x i32> %edgeCoord, i32 1, i32 16, <16 x i8> %call4)
  call spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiis(ptr addrspace(1) %image_out, <2 x i32> %edgeCoord, i32 1, i32 16, i16 %call5)
  call spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiiDv2_s(ptr addrspace(1) %image_out, <2 x i32> %edgeCoord, i32 1, i32 16, <2 x i16> %call6)
  call spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiiDv4_s(ptr addrspace(1) %image_out, <2 x i32> %edgeCoord, i32 1, i32 16, <4 x i16> %call7)
  call spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiiDv8_s(ptr addrspace(1) %image_out, <2 x i32> %edgeCoord, i32 1, i32 16, <8 x i16> %call8)
  call spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiiDv16_s(ptr addrspace(1) %image_out, <2 x i32> %edgeCoord, i32 1, i32 16, <16 x i16> %call9)
  call spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiii(ptr addrspace(1) %image_out, <2 x i32> %edgeCoord, i32 1, i32 16, i32 %call10)
  call spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiiS2_(ptr addrspace(1) %image_out, <2 x i32> %edgeCoord, i32 1, i32 16, <2 x i32> %call11)
  call spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiiDv4_i(ptr addrspace(1) %image_out, <2 x i32> %edgeCoord, i32 1, i32 16, <4 x i32> %call12)
  call spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiiDv8_i(ptr addrspace(1) %image_out, <2 x i32> %edgeCoord, i32 1, i32 16, <8 x i32> %call13)
  ret void
}

declare spir_func i8 @_Z46__spirv_SubgroupImageMediaBlockReadINTEL_RcharPU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(ptr addrspace(1), <2 x i32>, i32, i32)

declare spir_func <2 x i8> @_Z47__spirv_SubgroupImageMediaBlockReadINTEL_Rchar2PU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(ptr addrspace(1), <2 x i32>, i32, i32)

declare spir_func <4 x i8> @_Z47__spirv_SubgroupImageMediaBlockReadINTEL_Rchar4PU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(ptr addrspace(1), <2 x i32>, i32, i32)

declare spir_func <8 x i8> @_Z47__spirv_SubgroupImageMediaBlockReadINTEL_Rchar8PU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(ptr addrspace(1), <2 x i32>, i32, i32)

declare spir_func <16 x i8> @_Z48__spirv_SubgroupImageMediaBlockReadINTEL_Rchar16PU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(ptr addrspace(1), <2 x i32>, i32, i32)

declare spir_func i16 @_Z47__spirv_SubgroupImageMediaBlockReadINTEL_RshortPU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(ptr addrspace(1), <2 x i32>, i32, i32)

declare spir_func <2 x i16> @_Z48__spirv_SubgroupImageMediaBlockReadINTEL_Rshort2PU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(ptr addrspace(1), <2 x i32>, i32, i32)

declare spir_func <4 x i16> @_Z48__spirv_SubgroupImageMediaBlockReadINTEL_Rshort4PU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(ptr addrspace(1), <2 x i32>, i32, i32)

declare spir_func <8 x i16> @_Z48__spirv_SubgroupImageMediaBlockReadINTEL_Rshort8PU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(ptr addrspace(1), <2 x i32>, i32, i32)

declare spir_func <16 x i16> @_Z49__spirv_SubgroupImageMediaBlockReadINTEL_Rshort16PU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(ptr addrspace(1), <2 x i32>, i32, i32)

declare spir_func i32 @_Z45__spirv_SubgroupImageMediaBlockReadINTEL_RintPU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(ptr addrspace(1), <2 x i32>, i32, i32)

declare spir_func <2 x i32> @_Z46__spirv_SubgroupImageMediaBlockReadINTEL_Rint2PU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(ptr addrspace(1), <2 x i32>, i32, i32)

declare spir_func <4 x i32> @_Z46__spirv_SubgroupImageMediaBlockReadINTEL_Rint4PU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(ptr addrspace(1), <2 x i32>, i32, i32)

declare spir_func <8 x i32> @_Z46__spirv_SubgroupImageMediaBlockReadINTEL_Rint8PU3AS133__spirv_Image__void_1_0_0_0_0_0_0Dv2_iii(ptr addrspace(1), <2 x i32>, i32, i32)

declare spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiic(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1), <2 x i32>, i32, i32, i8)

declare spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiiDv2_c(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1), <2 x i32>, i32, i32, <2 x i8>)

declare spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiiDv4_c(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1), <2 x i32>, i32, i32, <4 x i8>)

declare spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiiDv8_c(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1), <2 x i32>, i32, i32, <8 x i8>)

declare spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiiDv16_c(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1), <2 x i32>, i32, i32, <16 x i8>)

declare spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiis(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1), <2 x i32>, i32, i32, i16)

declare spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiiDv2_s(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1), <2 x i32>, i32, i32, <2 x i16>)

declare spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiiDv4_s(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1), <2 x i32>, i32, i32, <4 x i16>)

declare spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiiDv8_s(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1), <2 x i32>, i32, i32, <8 x i16>)

declare spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiiDv16_s(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1), <2 x i32>, i32, i32, <16 x i16>)

declare spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiii(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1), <2 x i32>, i32, i32, i32)

declare spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiiS2_(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1), <2 x i32>, i32, i32, <2 x i32>)

declare spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiiDv4_i(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1), <2 x i32>, i32, i32, <4 x i32>)

declare spir_func void @_Z41__spirv_SubgroupImageMediaBlockWriteINTELPU3AS133__spirv_Image__void_1_0_0_0_0_0_1Dv2_iiiDv8_i(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1), <2 x i32>, i32, i32, <8 x i32>)

!spirv.MemoryModel = !{!0}
!opencl.enable.FP_CONTRACT = !{}
!spirv.Source = !{!1}
!opencl.spir.version = !{!2}
!opencl.ocl.version = !{!2}
!opencl.used.extensions = !{!3}
!opencl.used.optional.core.features = !{!4}
!spirv.Generator = !{!5}

!0 = !{i32 1, i32 2}
!1 = !{i32 3, i32 200000}
!2 = !{i32 2, i32 0}
!3 = !{}
!4 = !{!"cl_images"}
!5 = !{i16 6, i16 14}
!6 = !{i32 0, i32 1, i32 1}
!7 = !{!"none", !"read_only", !"write_only"}
!8 = !{!"int2", !"image2d_t", !"image2d_t"}

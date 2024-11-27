; Compiled from https://github.com/KhronosGroup/SPIRV-LLVM-Translator/test/extensions/INTEL/SPV_INTEL_media_block_io/SPV_INTEL_media_block_io.cl

; RUN: not llc -O0 -mtriple=spirv32-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; CHECK-ERROR: LLVM ERROR: intel_sub_group_media_block_read_uc: the builtin requires the following SPIR-V extension: SPV_INTEL_media_block_io

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_media_block_io %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_media_block_io %s -o - -filetype=obj | spirv-val %}
; CHECK: Capability SubgroupImageMediaBlockIOINTEL
; CHECK: Extension "SPV_INTEL_media_block_io"

; CHECK-COUNT-14: SubgroupImageMediaBlockReadINTEL
; CHECK-COUNT-14: SubgroupImageMediaBlockWriteINTEL

; ModuleID = '/localdisk2/vmaksimo/llvm-project/build/projects/SPIRV-LLVM-Translator/test/test_output/extensions/INTEL/SPV_INTEL_media_block_io/Output/SPV_INTEL_media_block_io.cl.tmp.bc'
source_filename = "/localdisk2/vmaksimo/llvm-project/llvm/projects/SPIRV-LLVM-Translator/test/extensions/INTEL/SPV_INTEL_media_block_io/SPV_INTEL_media_block_io.cl"
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spir-unknown-unknown"

; Function Attrs: convergent norecurse nounwind
define dso_local spir_kernel void @intel_media_block_test(<2 x i32> noundef %edgeCoord, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %src_luma_image, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1) %dst_luma_image) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %call = tail call spir_func zeroext i8 @_Z35intel_sub_group_media_block_read_ucDv2_iii14ocl_image2d_ro(<2 x i32> noundef %edgeCoord, i32 noundef 1, i32 noundef 16, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %src_luma_image) #2
  %call1 = tail call spir_func <2 x i8> @_Z36intel_sub_group_media_block_read_uc2Dv2_iii14ocl_image2d_ro(<2 x i32> noundef %edgeCoord, i32 noundef 1, i32 noundef 16, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %src_luma_image) #2
  %call2 = tail call spir_func <4 x i8> @_Z36intel_sub_group_media_block_read_uc4Dv2_iii14ocl_image2d_ro(<2 x i32> noundef %edgeCoord, i32 noundef 1, i32 noundef 16, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %src_luma_image) #2
  %call3 = tail call spir_func <8 x i8> @_Z36intel_sub_group_media_block_read_uc8Dv2_iii14ocl_image2d_ro(<2 x i32> noundef %edgeCoord, i32 noundef 1, i32 noundef 16, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %src_luma_image) #2
  %call4 = tail call spir_func <16 x i8> @_Z37intel_sub_group_media_block_read_uc16Dv2_iii14ocl_image2d_ro(<2 x i32> noundef %edgeCoord, i32 noundef 1, i32 noundef 16, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %src_luma_image) #2
  %call5 = tail call spir_func zeroext i16 @_Z35intel_sub_group_media_block_read_usDv2_iii14ocl_image2d_ro(<2 x i32> noundef %edgeCoord, i32 noundef 1, i32 noundef 16, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %src_luma_image) #2
  %call6 = tail call spir_func <2 x i16> @_Z36intel_sub_group_media_block_read_us2Dv2_iii14ocl_image2d_ro(<2 x i32> noundef %edgeCoord, i32 noundef 1, i32 noundef 16, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %src_luma_image) #2
  %call7 = tail call spir_func <4 x i16> @_Z36intel_sub_group_media_block_read_us4Dv2_iii14ocl_image2d_ro(<2 x i32> noundef %edgeCoord, i32 noundef 1, i32 noundef 16, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %src_luma_image) #2
  %call8 = tail call spir_func <8 x i16> @_Z36intel_sub_group_media_block_read_us8Dv2_iii14ocl_image2d_ro(<2 x i32> noundef %edgeCoord, i32 noundef 1, i32 noundef 16, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %src_luma_image) #2
  %call9 = tail call spir_func <16 x i16> @_Z37intel_sub_group_media_block_read_us16Dv2_iii14ocl_image2d_ro(<2 x i32> noundef %edgeCoord, i32 noundef 1, i32 noundef 16, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %src_luma_image) #2
  %call10 = tail call spir_func i32 @_Z35intel_sub_group_media_block_read_uiDv2_iii14ocl_image2d_ro(<2 x i32> noundef %edgeCoord, i32 noundef 1, i32 noundef 16, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %src_luma_image) #2
  %call11 = tail call spir_func <2 x i32> @_Z36intel_sub_group_media_block_read_ui2Dv2_iii14ocl_image2d_ro(<2 x i32> noundef %edgeCoord, i32 noundef 1, i32 noundef 16, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %src_luma_image) #2
  %call12 = tail call spir_func <4 x i32> @_Z36intel_sub_group_media_block_read_ui4Dv2_iii14ocl_image2d_ro(<2 x i32> noundef %edgeCoord, i32 noundef 1, i32 noundef 16, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %src_luma_image) #2
  %call13 = tail call spir_func <8 x i32> @_Z36intel_sub_group_media_block_read_ui8Dv2_iii14ocl_image2d_ro(<2 x i32> noundef %edgeCoord, i32 noundef 1, i32 noundef 16, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %src_luma_image) #2
  tail call spir_func void @_Z36intel_sub_group_media_block_write_ucDv2_iiih14ocl_image2d_wo(<2 x i32> noundef %edgeCoord, i32 noundef 1, i32 noundef 16, i8 noundef zeroext %call, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1) %dst_luma_image) #2
  tail call spir_func void @_Z37intel_sub_group_media_block_write_uc2Dv2_iiiDv2_h14ocl_image2d_wo(<2 x i32> noundef %edgeCoord, i32 noundef 1, i32 noundef 16, <2 x i8> noundef %call1, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1) %dst_luma_image) #2
  tail call spir_func void @_Z37intel_sub_group_media_block_write_uc4Dv2_iiiDv4_h14ocl_image2d_wo(<2 x i32> noundef %edgeCoord, i32 noundef 1, i32 noundef 16, <4 x i8> noundef %call2, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1) %dst_luma_image) #2
  tail call spir_func void @_Z37intel_sub_group_media_block_write_uc8Dv2_iiiDv8_h14ocl_image2d_wo(<2 x i32> noundef %edgeCoord, i32 noundef 1, i32 noundef 16, <8 x i8> noundef %call3, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1) %dst_luma_image) #2
  tail call spir_func void @_Z38intel_sub_group_media_block_write_uc16Dv2_iiiDv16_h14ocl_image2d_wo(<2 x i32> noundef %edgeCoord, i32 noundef 1, i32 noundef 16, <16 x i8> noundef %call4, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1) %dst_luma_image) #2
  tail call spir_func void @_Z36intel_sub_group_media_block_write_usDv2_iiit14ocl_image2d_wo(<2 x i32> noundef %edgeCoord, i32 noundef 1, i32 noundef 16, i16 noundef zeroext %call5, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1) %dst_luma_image) #2
  tail call spir_func void @_Z37intel_sub_group_media_block_write_us2Dv2_iiiDv2_t14ocl_image2d_wo(<2 x i32> noundef %edgeCoord, i32 noundef 1, i32 noundef 16, <2 x i16> noundef %call6, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1) %dst_luma_image) #2
  tail call spir_func void @_Z37intel_sub_group_media_block_write_us4Dv2_iiiDv4_t14ocl_image2d_wo(<2 x i32> noundef %edgeCoord, i32 noundef 1, i32 noundef 16, <4 x i16> noundef %call7, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1) %dst_luma_image) #2
  tail call spir_func void @_Z37intel_sub_group_media_block_write_us8Dv2_iiiDv8_t14ocl_image2d_wo(<2 x i32> noundef %edgeCoord, i32 noundef 1, i32 noundef 16, <8 x i16> noundef %call8, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1) %dst_luma_image) #2
  tail call spir_func void @_Z38intel_sub_group_media_block_write_us16Dv2_iiiDv16_t14ocl_image2d_wo(<2 x i32> noundef %edgeCoord, i32 noundef 1, i32 noundef 16, <16 x i16> noundef %call9, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1) %dst_luma_image) #2
  tail call spir_func void @_Z36intel_sub_group_media_block_write_uiDv2_iiij14ocl_image2d_wo(<2 x i32> noundef %edgeCoord, i32 noundef 1, i32 noundef 16, i32 noundef %call10, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1) %dst_luma_image) #2
  tail call spir_func void @_Z37intel_sub_group_media_block_write_ui2Dv2_iiiDv2_j14ocl_image2d_wo(<2 x i32> noundef %edgeCoord, i32 noundef 1, i32 noundef 16, <2 x i32> noundef %call11, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1) %dst_luma_image) #2
  tail call spir_func void @_Z37intel_sub_group_media_block_write_ui4Dv2_iiiDv4_j14ocl_image2d_wo(<2 x i32> noundef %edgeCoord, i32 noundef 1, i32 noundef 16, <4 x i32> noundef %call12, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1) %dst_luma_image) #2
  tail call spir_func void @_Z37intel_sub_group_media_block_write_ui8Dv2_iiiDv8_j14ocl_image2d_wo(<2 x i32> noundef %edgeCoord, i32 noundef 1, i32 noundef 16, <8 x i32> noundef %call13, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1) %dst_luma_image) #2
  ret void
}

; Function Attrs: convergent nounwind
declare spir_func zeroext i8 @_Z35intel_sub_group_media_block_read_ucDv2_iii14ocl_image2d_ro(<2 x i32> noundef, i32 noundef, i32 noundef, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0)) local_unnamed_addr #1

; Function Attrs: convergent nounwind
declare spir_func <2 x i8> @_Z36intel_sub_group_media_block_read_uc2Dv2_iii14ocl_image2d_ro(<2 x i32> noundef, i32 noundef, i32 noundef, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0)) local_unnamed_addr #1

; Function Attrs: convergent nounwind
declare spir_func <4 x i8> @_Z36intel_sub_group_media_block_read_uc4Dv2_iii14ocl_image2d_ro(<2 x i32> noundef, i32 noundef, i32 noundef, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0)) local_unnamed_addr #1

; Function Attrs: convergent nounwind
declare spir_func <8 x i8> @_Z36intel_sub_group_media_block_read_uc8Dv2_iii14ocl_image2d_ro(<2 x i32> noundef, i32 noundef, i32 noundef, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0)) local_unnamed_addr #1

; Function Attrs: convergent nounwind
declare spir_func <16 x i8> @_Z37intel_sub_group_media_block_read_uc16Dv2_iii14ocl_image2d_ro(<2 x i32> noundef, i32 noundef, i32 noundef, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0)) local_unnamed_addr #1

; Function Attrs: convergent nounwind
declare spir_func zeroext i16 @_Z35intel_sub_group_media_block_read_usDv2_iii14ocl_image2d_ro(<2 x i32> noundef, i32 noundef, i32 noundef, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0)) local_unnamed_addr #1

; Function Attrs: convergent nounwind
declare spir_func <2 x i16> @_Z36intel_sub_group_media_block_read_us2Dv2_iii14ocl_image2d_ro(<2 x i32> noundef, i32 noundef, i32 noundef, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0)) local_unnamed_addr #1

; Function Attrs: convergent nounwind
declare spir_func <4 x i16> @_Z36intel_sub_group_media_block_read_us4Dv2_iii14ocl_image2d_ro(<2 x i32> noundef, i32 noundef, i32 noundef, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0)) local_unnamed_addr #1

; Function Attrs: convergent nounwind
declare spir_func <8 x i16> @_Z36intel_sub_group_media_block_read_us8Dv2_iii14ocl_image2d_ro(<2 x i32> noundef, i32 noundef, i32 noundef, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0)) local_unnamed_addr #1

; Function Attrs: convergent nounwind
declare spir_func <16 x i16> @_Z37intel_sub_group_media_block_read_us16Dv2_iii14ocl_image2d_ro(<2 x i32> noundef, i32 noundef, i32 noundef, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0)) local_unnamed_addr #1

; Function Attrs: convergent nounwind
declare spir_func i32 @_Z35intel_sub_group_media_block_read_uiDv2_iii14ocl_image2d_ro(<2 x i32> noundef, i32 noundef, i32 noundef, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0)) local_unnamed_addr #1

; Function Attrs: convergent nounwind
declare spir_func <2 x i32> @_Z36intel_sub_group_media_block_read_ui2Dv2_iii14ocl_image2d_ro(<2 x i32> noundef, i32 noundef, i32 noundef, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0)) local_unnamed_addr #1

; Function Attrs: convergent nounwind
declare spir_func <4 x i32> @_Z36intel_sub_group_media_block_read_ui4Dv2_iii14ocl_image2d_ro(<2 x i32> noundef, i32 noundef, i32 noundef, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0)) local_unnamed_addr #1

; Function Attrs: convergent nounwind
declare spir_func <8 x i32> @_Z36intel_sub_group_media_block_read_ui8Dv2_iii14ocl_image2d_ro(<2 x i32> noundef, i32 noundef, i32 noundef, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0)) local_unnamed_addr #1

; Function Attrs: convergent nounwind
declare spir_func void @_Z36intel_sub_group_media_block_write_ucDv2_iiih14ocl_image2d_wo(<2 x i32> noundef, i32 noundef, i32 noundef, i8 noundef zeroext, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1)) local_unnamed_addr #1

; Function Attrs: convergent nounwind
declare spir_func void @_Z37intel_sub_group_media_block_write_uc2Dv2_iiiDv2_h14ocl_image2d_wo(<2 x i32> noundef, i32 noundef, i32 noundef, <2 x i8> noundef, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1)) local_unnamed_addr #1

; Function Attrs: convergent nounwind
declare spir_func void @_Z37intel_sub_group_media_block_write_uc4Dv2_iiiDv4_h14ocl_image2d_wo(<2 x i32> noundef, i32 noundef, i32 noundef, <4 x i8> noundef, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1)) local_unnamed_addr #1

; Function Attrs: convergent nounwind
declare spir_func void @_Z37intel_sub_group_media_block_write_uc8Dv2_iiiDv8_h14ocl_image2d_wo(<2 x i32> noundef, i32 noundef, i32 noundef, <8 x i8> noundef, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1)) local_unnamed_addr #1

; Function Attrs: convergent nounwind
declare spir_func void @_Z38intel_sub_group_media_block_write_uc16Dv2_iiiDv16_h14ocl_image2d_wo(<2 x i32> noundef, i32 noundef, i32 noundef, <16 x i8> noundef, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1)) local_unnamed_addr #1

; Function Attrs: convergent nounwind
declare spir_func void @_Z36intel_sub_group_media_block_write_usDv2_iiit14ocl_image2d_wo(<2 x i32> noundef, i32 noundef, i32 noundef, i16 noundef zeroext, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1)) local_unnamed_addr #1

; Function Attrs: convergent nounwind
declare spir_func void @_Z37intel_sub_group_media_block_write_us2Dv2_iiiDv2_t14ocl_image2d_wo(<2 x i32> noundef, i32 noundef, i32 noundef, <2 x i16> noundef, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1)) local_unnamed_addr #1

; Function Attrs: convergent nounwind
declare spir_func void @_Z37intel_sub_group_media_block_write_us4Dv2_iiiDv4_t14ocl_image2d_wo(<2 x i32> noundef, i32 noundef, i32 noundef, <4 x i16> noundef, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1)) local_unnamed_addr #1

; Function Attrs: convergent nounwind
declare spir_func void @_Z37intel_sub_group_media_block_write_us8Dv2_iiiDv8_t14ocl_image2d_wo(<2 x i32> noundef, i32 noundef, i32 noundef, <8 x i16> noundef, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1)) local_unnamed_addr #1

; Function Attrs: convergent nounwind
declare spir_func void @_Z38intel_sub_group_media_block_write_us16Dv2_iiiDv16_t14ocl_image2d_wo(<2 x i32> noundef, i32 noundef, i32 noundef, <16 x i16> noundef, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1)) local_unnamed_addr #1

; Function Attrs: convergent nounwind
declare spir_func void @_Z36intel_sub_group_media_block_write_uiDv2_iiij14ocl_image2d_wo(<2 x i32> noundef, i32 noundef, i32 noundef, i32 noundef, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1)) local_unnamed_addr #1

; Function Attrs: convergent nounwind
declare spir_func void @_Z37intel_sub_group_media_block_write_ui2Dv2_iiiDv2_j14ocl_image2d_wo(<2 x i32> noundef, i32 noundef, i32 noundef, <2 x i32> noundef, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1)) local_unnamed_addr #1

; Function Attrs: convergent nounwind
declare spir_func void @_Z37intel_sub_group_media_block_write_ui4Dv2_iiiDv4_j14ocl_image2d_wo(<2 x i32> noundef, i32 noundef, i32 noundef, <4 x i32> noundef, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1)) local_unnamed_addr #1

; Function Attrs: convergent nounwind
declare spir_func void @_Z37intel_sub_group_media_block_write_ui8Dv2_iiiDv8_j14ocl_image2d_wo(<2 x i32> noundef, i32 noundef, i32 noundef, <8 x i32> noundef, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1)) local_unnamed_addr #1

attributes #0 = { convergent norecurse nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" }
attributes #1 = { convergent nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent nounwind }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{!"clang version 20.0.0git (https://github.com/llvm/llvm-project.git 32da1fd8c7d45d5209c6c781910c51940779ec52)"}
!3 = !{i32 0, i32 1, i32 1}
!4 = !{!"none", !"read_only", !"write_only"}
!5 = !{!"int2", !"image2d_t", !"image2d_t"}
!6 = !{!"int __attribute__((ext_vector_type(2)))", !"image2d_t", !"image2d_t"}
!7 = !{!"", !"", !""}

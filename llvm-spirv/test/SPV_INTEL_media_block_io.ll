; Source
;__kernel void vme_kernel(int2 edgeCoord, __read_only image2d_t src_luma_image,
;                         __write_only image2d_t dst_luma_image) {
;  uint Edge =
;      intel_sub_group_media_block_read_ui(edgeCoord, 1, 16, src_luma_image);
;  // make sure that we can translate functions which differ only by return type
;  uchar Edge2 =
;      intel_sub_group_media_block_read_uc(edgeCoord, 1, 16, src_luma_image);
;
;  intel_sub_group_media_block_write_ui(edgeCoord, 1, 16, Edge, dst_luma_image);
;}

; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s

; CHECK: Capability SubgroupImageMediaBlockIOINTEL

; CHECK: TypeInt [[TypeInt:[0-9]+]] 32
; CHECK: TypeInt [[TypeChar:[0-9]+]] 8
; CHECK: Constant [[TypeInt]] [[One:[0-9]+]] 1
; CHECK: Constant [[TypeInt]] [[Sixteen:[0-9]+]] 16

; CHECK: FunctionParameter {{[0-9]+}} [[Coord:[0-9]+]]
; CHECK-NEXT: FunctionParameter {{[0-9]+}} [[SrcImage:[0-9]+]]
; CHECK-NEXT: FunctionParameter {{[0-9]+}} [[DstImage:[0-9]+]]

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

%opencl.image2d_ro_t = type opaque
%opencl.image2d_wo_t = type opaque

; Function Attrs: nounwind
define spir_kernel void @vme_kernel(<2 x i32> %edgeCoord, %opencl.image2d_ro_t addrspace(1)* %src_luma_image, %opencl.image2d_wo_t addrspace(1)* %dst_luma_image) local_unnamed_addr #0 !kernel_arg_addr_space !5 !kernel_arg_access_qual !6 !kernel_arg_type !7 !kernel_arg_base_type !8 !kernel_arg_type_qual !9 {
entry:
; CHECK: SubgroupImageMediaBlockReadINTEL [[TypeInt]] [[Data:[0-9]+]] [[SrcImage]] [[Coord]] [[One]] [[Sixteen]]
  %call = tail call spir_func i32 @_Z35intel_sub_group_media_block_read_uiDv2_iii14ocl_image2d_ro(<2 x i32> %edgeCoord, i32 1, i32 16, %opencl.image2d_ro_t addrspace(1)* %src_luma_image) #2
; CHECK: SubgroupImageMediaBlockReadINTEL [[TypeChar]] {{[0-9]+}} [[SrcImage]] [[Coord]] [[One]] [[Sixteen]]
  %call1 = tail call spir_func zeroext i8 @_Z35intel_sub_group_media_block_read_ucDv2_iii14ocl_image2d_ro(<2 x i32> %edgeCoord, i32 1, i32 16, %opencl.image2d_ro_t addrspace(1)* %src_luma_image) #2
; CHECK: SubgroupImageMediaBlockWriteINTEL [[DstImage]] [[Coord]] [[One]] [[Sixteen]] [[Data]]
  tail call spir_func void @_Z36intel_sub_group_media_block_write_uiDv2_iiij14ocl_image2d_wo(<2 x i32> %edgeCoord, i32 1, i32 16, i32 %call, %opencl.image2d_wo_t addrspace(1)* %dst_luma_image) #2
  ret void
}

declare spir_func i32 @_Z35intel_sub_group_media_block_read_uiDv2_iii14ocl_image2d_ro(<2 x i32>, i32, i32, %opencl.image2d_ro_t addrspace(1)*) local_unnamed_addr #1

declare spir_func zeroext i8 @_Z35intel_sub_group_media_block_read_ucDv2_iii14ocl_image2d_ro(<2 x i32>, i32, i32, %opencl.image2d_ro_t addrspace(1)*) local_unnamed_addr #1

declare spir_func void @_Z36intel_sub_group_media_block_write_uiDv2_iiij14ocl_image2d_wo(<2 x i32>, i32, i32, i32, %opencl.image2d_wo_t addrspace(1)*) local_unnamed_addr #1

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0}
!opencl.enable.FP_CONTRACT = !{}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!3}
!opencl.compiler.options = !{!2}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{}
!3 = !{!"cl_images"}
!4 = !{!"clang version 5.0.1 (cfe/trunk)"}
!5 = !{i32 0, i32 1, i32 1}
!6 = !{!"none", !"read_only", !"write_only"}
!7 = !{!"int2", !"image2d_t", !"image2d_t"}
!8 = !{!"int __attribute__((ext_vector_type(2)))", !"image2d_t", !"image2d_t"}
!9 = !{!"", !"", !""}

; RUN: llvm-as < %s | llvm-spirv -spirv-text -o %t
; RUN: FileCheck < %t %s
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

%opencl.image2d_t = type opaque

; CHECK: 8 Decorate [[ID:[0-9]*]] LinkageAttributes "imageSampler" Export
; CHECK: 5 Variable {{[0-9]*}} [[ID]] 0 {{[0-9]*}}

@imageSampler = addrspace(2) constant i32 36, align 4

; Function Attrs: nounwind
define spir_kernel void @sample_kernel(%opencl.image2d_t addrspace(1)* %input, float addrspace(1)* nocapture %xOffsets, float addrspace(1)* nocapture %yOffsets, <4 x float> addrspace(1)* nocapture %results) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !5 !kernel_arg_type_qual !4 {
  %1 = tail call spir_func i64 @_Z13get_global_idj(i32 0) #1
  %2 = trunc i64 %1 to i32
  %3 = tail call spir_func i64 @_Z13get_global_idj(i32 1) #1
  %4 = trunc i64 %3 to i32
  %5 = tail call spir_func i32 @_Z15get_image_width11ocl_image2d(%opencl.image2d_t addrspace(1)* %input) #1
  %6 = mul nsw i32 %4, %5
  %7 = add nsw i32 %6, %2
  %8 = sitofp i32 %2 to float
  %9 = insertelement <2 x float> undef, float %8, i32 0
  %10 = sitofp i32 %4 to float
  %11 = insertelement <2 x float> %9, float %10, i32 1
  %12 = tail call spir_func <4 x float> @_Z11read_imagef11ocl_image2d11ocl_samplerDv2_f(%opencl.image2d_t addrspace(1)* %input, i32 36, <2 x float> %11) #1
  %13 = sext i32 %7 to i64
  %14 = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %results, i64 %13
  store <4 x float> %12, <4 x float> addrspace(1)* %14, align 16, !tbaa !11
  ret void
}

; Function Attrs: nounwind readnone
declare spir_func i64 @_Z13get_global_idj(i32) #1

; Function Attrs: nounwind readnone
declare spir_func i32 @_Z15get_image_width11ocl_image2d(%opencl.image2d_t addrspace(1)*) #1

; Function Attrs: nounwind readnone
declare spir_func <4 x float> @_Z11read_imagef11ocl_image2d11ocl_samplerDv2_f(%opencl.image2d_t addrspace(1)*, i32, <2 x float>) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!7}
!opencl.ocl.version = !{!7}
!opencl.used.extensions = !{!8}
!opencl.used.optional.core.features = !{!9}
!opencl.compiler.options = !{!10}

!1 = !{i32 1, i32 1, i32 1, i32 1}
!2 = !{!"read_only", !"none", !"none", !"none"}
!3 = !{!"image2d_float", !"float*", !"float*", !"float4*"}
!4 = !{!"", !"", !"", !""}
!5 = !{!"image2d_t", !"float*", !"float*", !"float4*"}
!7 = !{i32 1, i32 2}
!8 = !{}
!9 = !{!"cl_images"}
!10 = !{!"-cl-kernel-arg-info"}
!11 = !{!12, !12, i64 0}
!12 = !{!"omnipotent char", !13}
!13 = !{!"Simple C/C++ TBAA"}

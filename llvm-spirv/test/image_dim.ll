; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV-DAG: 2 Capability Sampled1D
; CHECK-SPIRV-DAG: 2 Capability SampledBuffer

; ModuleID = 'image_d_20.bc'
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

%opencl.image1d_t = type opaque
%opencl.image1d_buffer_t = type opaque

; Function Attrs: nounwind
define spir_kernel void @image_d(%opencl.image1d_t addrspace(1)* %image1d_td6, %opencl.image1d_buffer_t addrspace(1)* %image1d_buffer_td8) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
entry:
  ret void
}

attributes #0 = { nounwind }

!opencl.enable.FP_CONTRACT = !{}
!spirv.Source = !{!6}
!opencl.spir.version = !{!7}
!opencl.ocl.version = !{!8}
!opencl.used.extensions = !{!9}
!opencl.used.optional.core.features = !{!10}
!spirv.Generator = !{!11}

!1 = !{i32 1, i32 1}
!2 = !{!"read_only", !"read_only"}
!3 = !{!"image1d_t", !"image1d_buffer_t"}
!4 = !{!"", !""}
!5 = !{!"image1d_t", !"image1d_buffer_t"}
!6 = !{i32 3, i32 100000}
!7 = !{i32 1, i32 2}
!8 = !{i32 1, i32 0}
!9 = !{}
!10 = !{!"cl_images"}
!11 = !{i16 6, i16 14}

;__kernel void sample_test(read_only image2d_t src, read_only image1d_buffer_t buff){}

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV-NOT: 2 Capability Shader

; ModuleID = 'test.bc'
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%opencl.image2d_ro_t = type opaque
%opencl.image1d_buffer_ro_t = type opaque

; Function Attrs: nounwind
define spir_kernel void @sample_test(%opencl.image2d_ro_t addrspace(1)* %src, %opencl.image1d_buffer_ro_t addrspace(1)* %buf) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
entry:
  ret void
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!6}
!opencl.ocl.version = !{!7}
!opencl.used.extensions = !{!8}
!opencl.used.optional.core.features = !{!8}
!opencl.compiler.options = !{!8}
!llvm.ident = !{!9}

!1 = !{i32 1, i32 1}
!2 = !{!"read_only", !"read_only"}
!3 = !{!"__read_only image2d_t", !"__read_only image1d_buffer_t"}
!4 = !{!"__read_only image2d_t", !"__read_only image1d_buffer_t"}
!5 = !{!"", !""}
!6 = !{i32 1, i32 2}
!7 = !{i32 2, i32 0}
!8 = !{}
!9 = !{!"clang version 3.6.1"}

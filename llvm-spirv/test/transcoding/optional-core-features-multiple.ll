; OpenCL C source
; -----------------------------------------------
; double d = 1.0;
; kernel void test(read_only image2d_t img) {}
; -----------------------------------------------
;
; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

%opencl.image2d_t = type opaque

@d = addrspace(1) global double 1.000000e+00, align 8

; Function Attrs: nounwind readnone
define spir_kernel void @test(%opencl.image2d_t addrspace(1)* nocapture %img) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
entry:
  ret void
}

attributes #0 = { nounwind readnone }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!6}
!opencl.ocl.version = !{!7}
!opencl.used.extensions = !{!8}
!opencl.used.optional.core.features = !{!9}
!opencl.compiler.options = !{!8}

!1 = !{i32 1}
!2 = !{!"read_only"}
!3 = !{!"image2d_t"}
!4 = !{!"image2d_t"}
!5 = !{!""}
!6 = !{i32 1, i32 2}
!7 = !{i32 2, i32 0}
!8 = !{}
!9 = !{!"cl_doubles", !"cl_images"}
; CHECK-SPIRV: 2 Capability Float64
; CHECK-SPIRV: 2 Capability ImageBasic
; CHECK-LLVM: {{.*}} !{!"cl_doubles", !"cl_images"}
; CHECK-LLVM-NOT: {{.*}} !{!"cl_doubles cl_images"}

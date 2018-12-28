; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV: {{[0-9]*}} TypeImage [[TypeImage:[0-9]+]]
; CHECK-SPIRV-NOT: {{[0-9]*}} TypeImage
; CHECK-SPIRV: {{[0-9]*}} TypeFunction {{[0-9]*}} {{[0-9]*}} [[TypeImage]]
; CHECK-SPIRV: {{[0-9]*}} TypeFunction {{[0-9]*}} {{[0-9]*}} [[TypeImage]]
; CHECK-SPIRV: {{[0-9]*}} FunctionParameter [[TypeImage]] {{[0-9]*}}
; CHECK-SPIRV: {{[0-9]*}} FunctionParameter [[TypeImage]] {{[0-9]*}}
; CHECK-SPIRV: {{[0-9]*}} FunctionParameter [[TypeImage]] [[ParamID:[0-9]+]]
; CHECK-SPIRV: {{[0-9]*}} FunctionCall {{[0-9]*}} {{[0-9]*}} {{[0-9]*}} [[ParamID]]

; ModuleID = 'test.cl.bc'
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

%opencl.image2d_ro_t = type opaque

; Function Attrs: nounwind
define spir_func void @f0(%opencl.image2d_ro_t addrspace(1)* %v2, <2 x float> %v3) #0 {
entry:
  ret void
}

; Function Attrs: nounwind
define spir_func void @f1(%opencl.image2d_ro_t addrspace(1)* %v2, <2 x float> %v3) #0 {
entry:
  ret void
}

; Function Attrs: nounwind
define spir_kernel void @test(%opencl.image2d_ro_t addrspace(1)* %v1) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !5 !kernel_arg_type_qual !4 {
entry:
  call spir_func void @f0(%opencl.image2d_ro_t addrspace(1)* %v1, <2 x float> <float 1.000000e+00, float 5.000000e+00>) #0
  ret void
}

attributes #0 = { nounwind }

!opencl.enable.FP_CONTRACT = !{}
!spirv.Source = !{!6}
!opencl.spir.version = !{!7}
!opencl.ocl.version = !{!7}
!opencl.used.extensions = !{!8}
!opencl.used.optional.core.features = !{!9}
!spirv.Generator = !{!10}

!1 = !{i32 1}
!2 = !{!"read_only"}
!3 = !{!"image2d_t"}
!4 = !{!""}
!5 = !{!"image2d_t"}
!6 = !{i32 3, i32 200000}
!7 = !{i32 2, i32 0}
!8 = !{}
!9 = !{!"cl_images"}
!10 = !{i16 6, i16 14}

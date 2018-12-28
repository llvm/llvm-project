; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-LLVM: call spir_func float @_Z5clampfff(
; CHECK-LLVM: call spir_func half @_Z5clampDhDhDh(

; CHECK-SPIRV: 8 ExtInst {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} fclamp
; CHECK-SPIRV-NOT: 8 ExtInst {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} clamp

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @test_scalar(float addrspace(1)* nocapture readonly %f) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
entry:
  %0 = load float, float addrspace(1)* %f, align 4
  %call = tail call spir_func float @_Z5clampfff(float %0, float 0.000000e+00, float 1.000000e+00) #2
  %1 = load float, float addrspace(1)* %f, align 4
  %conv = fptrunc float %1 to half
  %call1 = tail call spir_func half @_Z5clampDhDhDh(half %conv, half %conv, half %conv) #2
  ret void
}

declare spir_func float @_Z5clampfff(float, float, float) #1

declare spir_func half @_Z5clampDhDhDh(half, half, half) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!6}
!opencl.ocl.version = !{!7}
!opencl.used.extensions = !{!8}
!opencl.used.optional.core.features = !{!9}
!opencl.compiler.options = !{!9}

!1 = !{i32 1}
!2 = !{!"none"}
!3 = !{!"float*"}
!4 = !{!"float*"}
!5 = !{!""}
!6 = !{i32 1, i32 2}
!7 = !{i32 2, i32 0}
!8 = !{!"cl_khr_fp16"}
!9 = !{}

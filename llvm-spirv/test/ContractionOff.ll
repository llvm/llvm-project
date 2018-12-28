; Source:
; void kernel k1 (float a, float b, float c) {
; #pragma OPENCL FP_CONTRACT OFF
;   float d = a * b + c;
; }
;
; void kernel k2 (float a, float b, float c) {
;   float d = a * b + c;
; }

; RUN: llvm-as < %s | llvm-spirv -spirv-text -o %t
; RUN: FileCheck < %t %s

; CHECK: EntryPoint 6 [[K1:[0-9]+]] "k1"
; CHECK: EntryPoint 6 [[K2:[0-9]+]] "k2"
; CHECK: ExecutionMode [[K1]] 31
; CHECK-NOT: ExecutionMode [[K2]] 31

;ModuleID = '<stdin>'
source_filename = "/tmp/tmp.cl"
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

; Function Attrs: convergent nounwind
define spir_kernel void @k1(float %a, float %b, float %c) #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !6 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %mul = fmul float %a, %b
  %add = fadd float %mul, %c
  ret void
}

; Function Attrs: convergent nounwind
define spir_kernel void @k2(float %a, float %b, float %c) #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !6 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %0 = call float @llvm.fmuladd.f32(float %a, float %b, float %c)
  ret void
}

; Function Attrs: nounwind readnone speculatable
declare float @llvm.fmuladd.f32(float, float, float) #2

attributes #0 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind readnone speculatable }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 0}
!2 = !{i32 1, i32 2}
!3 = !{!"clang version 8.0.0"}
!4 = !{i32 0, i32 0, i32 0}
!5 = !{!"none", !"none", !"none"}
!6 = !{!"float", !"float", !"float"}
!7 = !{!"", !"", !""}

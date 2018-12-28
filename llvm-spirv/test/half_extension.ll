;#pragma OPENCL EXTENSION cl_khr_fp16 : enable
;half test()
;{
;   half x = 0.1f;
;   x+=2.0f;
;   half y = x + x;
;   return y;
;}

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv

; CHECK-SPIRV: {{[0-9]+}} Capability Float16

; ModuleID = 'main'
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
define spir_func half @test() #0 {
entry:
  %x = alloca half, align 2
  %y = alloca half, align 2
  store half 0xH2E66, half* %x, align 2
  %0 = load half, half* %x, align 2
  %conv = fpext half %0 to float
  %add = fadd float %conv, 2.000000e+00
  %conv1 = fptrunc float %add to half
  store half %conv1, half* %x, align 2
  %1 = load half, half* %x, align 2
  %2 = load half, half* %x, align 2
  %add2 = fadd half %1, %2
  store half %add2, half* %y, align 2
  %3 = load half, half* %y, align 2
  ret half %3
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!0}
!opencl.ocl.version = !{!0}
!opencl.used.extensions = !{!1}
!opencl.used.optional.core.features = !{!2}
!opencl.compiler.options = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, i32 2}
!1 = !{!"cl_khr_fp16"}
!2 = !{}
!3 = !{!"clang version 3.6.1"}

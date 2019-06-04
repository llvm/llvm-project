; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

; CHECK-NOT: llvm.fmuladd

; CHECK: TypeFloat [[f32:[0-9]+]] 32
; CHECK: TypeFloat [[f64:[0-9]+]] 64

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64"

; Function Attrs: nounwind
define spir_func void @foo(float %a, float %b, float %c, double %x, double %y, double %z) #0 {
entry:
  %0 = call float @llvm.fmuladd.f32(float %a, float %b, float %c)
; CHECK: FMul [[f32]] [[mul32:[0-9]+]] {{[0-9]+}} {{[0-9]+}}
; CHECK-NEXT: FAdd [[f32]] {{[0-9]+}} [[mul32]] {{[0-9]+}}
  %1 = call double @llvm.fmuladd.f64(double %x, double %y, double %z)
; CHECK: FMul [[f64]] [[mul64:[0-9]+]] {{[0-9]+}} {{[0-9]+}}
; CHECK-NEXT: FAdd [[f64]] {{[0-9]+}} [[mul64]] {{[0-9]+}}
ret void
}

; Function Attrs: nounwind readnone
declare float @llvm.fmuladd.f32(float, float, float) #1

; Function Attrs: nounwind readnone
declare double @llvm.fmuladd.f64(double, double, double) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!0}
!opencl.ocl.version = !{!1}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!3}
!opencl.compiler.options = !{!2}

!0 = !{i32 1, i32 2}
!1 = !{i32 2, i32 0}
!2 = !{}
!3 = !{!"cl_doubles"}

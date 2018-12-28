; Translator should not translate llvm intrinsic calls straight forward.
; It either represnts intrinsic's semantics with SPIRV instruction(s), or
; reports an error.
; XFAIL: *
; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64"

; Function Attrs: nounwind
define spir_func void @foo(float %a, float %b, float %c) #0 {
entry:
  %0 = call float @llvm.fma.f32(float %a, float %b, float %c)
  ret void
}

; Function Attrs: nounwind readnone
declare float @llvm.fma.f32(float, float, float) #1

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

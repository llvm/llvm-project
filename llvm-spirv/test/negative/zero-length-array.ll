; RUN: llvm-as %s -o %t.bc
; RUN: not --crash llvm-spirv %t.bc -o %t.spv 2>&1 | FileCheck %s

; CHECK: InvalidArraySize: Array size must be at least 1: [0 x i32]

source_filename = "test.cl"
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

; Function Attrs: convergent noinline nounwind optnone
define spir_func void @_Z3foov() #0 {
entry:
  %i = alloca [0 x i32], align 4
  ret void
}

attributes #0 = { convergent noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 0, i32 0}
!2 = !{i32 0, i32 2}
!3 = !{!"clang version 9.0.0"}

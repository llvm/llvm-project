;uint8 foo(uint8 c, unsigned i) {
;  c[i] = 42;
;  return c;
;}

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s

; CHECK: TypeInt [[TypeInt:[0-9]+]] 32
; CHECK: TypeVector [[TypeVector:[0-9]+]] [[TypeInt]] 8
; CHECK: VectorInsertDynamic [[TypeVector]] {{[0-9]+}}

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

; Function Attrs: norecurse nounwind readnone
define spir_func <8 x i32> @foo(<8 x i32> %c, i32 %i) local_unnamed_addr #0 {
entry:
  %vecins = insertelement <8 x i32> %c, i32 42, i32 %i
  ret <8 x i32> %vecins
}

attributes #0 = { norecurse nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!opencl.enable.FP_CONTRACT = !{}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!2}
!opencl.used.extensions = !{!3}
!opencl.used.optional.core.features = !{!3}
!opencl.compiler.options = !{!3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 0}
!2 = !{i32 1, i32 2}
!3 = !{}
!4 = !{!"clang version 6.0.0 (cfe/trunk)"}

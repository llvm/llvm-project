; Source
; int square(unsigned short a) {
;   return a * a;
; }
; Command
; clang -cc1 -triple spir -emit-llvm -O2 -o NoSignedUnsignedWrap.ll test.cl

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: Extension "SPV_KHR_no_integer_wrap_decoration"

; CHECK-SPIRV: Decorate {{[0-9]+}} NoSignedWrap
; CHECK-SPIRV: Decorate {{[0-9]+}} NoUnsignedWrap

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

; Function Attrs: norecurse nounwind readnone
define spir_func i32 @square(i16 zeroext %a) local_unnamed_addr #0 {
entry:
  %conv = zext i16 %a to i32
  ; CHECK-LLVM: mul nuw nsw
  %mul = mul nuw nsw i32 %conv, %conv
  ret i32 %mul
}

attributes #0 = { norecurse nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 0}
!2 = !{i32 1, i32 2}

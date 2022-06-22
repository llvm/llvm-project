; RUN: llc < %s | FileCheck %s --check-prefix=CHECK --check-prefix=SAFE
; RUN: llc < %s -enable-unsafe-fp-math | FileCheck %s --check-prefix=CHECK --check-prefix=UNSAFE

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64--"

; CHECK-LABEL: double_rounding:
; SAFE: callq __trunctfdf2
; SAFE-NEXT: cvtsd2ss %xmm0
; UNSAFE: callq __trunctfsf2
; UNSAFE-NOT: cvt
define void @double_rounding(ptr %x, ptr %f) {
entry:
  %0 = load fp128, ptr %x, align 16
  %1 = fptrunc fp128 %0 to double
  %2 = fptrunc double %1 to float
  store float %2, ptr %f, align 4
  ret void
}

; CHECK-LABEL: double_rounding_precise_first:
; CHECK: fstps (%
; CHECK-NOT: fstpl
define void @double_rounding_precise_first(ptr %f) {
entry:
  ; Hack, to generate a precise FP_ROUND to double
  %precise = call double asm sideeffect "fld %st(0)", "={st(0)}"()
  %0 = fptrunc double %precise to float
  store float %0, ptr %f, align 4
  ret void
}

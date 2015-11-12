; RUN: llc < %s -mtriple=x86_64-darwin | FileCheck %s

; CHECK-LABEL: LCPI0_0:
; CHECK-NEXT: .long 4286578688
; CHECK-LABEL: LCPI0_1:
; CHECK-NEXT: .long 2139095040

; CHECK-LABEL: foo:
; CHECK: testb $-128, -15(%rsp)
; CHECK: flds LCPI0_0(%rip)
; CHECK: flds LCPI0_1(%rip)
; CHECK: fcmovne %st(1), %st(0)
; CHECK: fstp %st(1)
; CHECK: retq
define x86_fp80 @foo(x86_fp80 %a) {
  %1 = tail call x86_fp80 @copysignl(x86_fp80 0xK7FFF8000000000000000, x86_fp80 %a) nounwind readnone
  ret x86_fp80 %1
}

declare x86_fp80 @copysignl(x86_fp80, x86_fp80) nounwind readnone

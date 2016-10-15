; RUN: llc -mtriple=arm64 -fp-contract=fast -o - %s | FileCheck %s


; Make sure we don't try to fold an fneg into +0.0, creating an illegal constant
; -0.0. It's also good, though not essential, that we don't resort to a litpool.
define double @test_fms_fold(double %a, double %b) {
; CHECK-LABEL: test_fms_fold:
; CHECK: fmov {{d[0-9]+}}, xzr
; CHECK: ret
  %mul = fmul double %a, 0.000000e+00
  %mul1 = fmul double %b, 0.000000e+00
  %sub = fsub double %mul, %mul1
  ret double %sub
}

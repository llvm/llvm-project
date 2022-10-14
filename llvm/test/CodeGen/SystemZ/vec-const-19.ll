; Test that a scalar FP constant can be reused from a vector splat constant
; of the same value.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

define void @fun()  {
; CHECK-LABEL: fun:
; CHECK: vgmg %v0, 2, 10
; CHECK-NOT: vgmg %v0, 2, 10

  %tmp = fadd <2 x double> zeroinitializer, <double 1.000000e+00, double 1.000000e+00>
  %tmp1 = fmul <2 x double> %tmp, <double 5.000000e-01, double 5.000000e-01>
  store <2 x double> %tmp1, ptr undef
  %tmp2 = load double, ptr undef
  %tmp3 = fmul double %tmp2, 5.000000e-01
  store double %tmp3, ptr undef
  ret void
}

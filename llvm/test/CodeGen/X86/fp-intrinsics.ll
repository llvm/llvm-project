; RUN: llc -O3 -mtriple=x86_64-pc-linux < %s | FileCheck %s

; Verify that constants aren't folded to inexact results when the rounding mode
; is unknown.
;
; double f1() {
;   // Because 0.1 cannot be represented exactly, this shouldn't be folded.
;   return 1.0/10.0;
; }
;
; CHECK-LABEL: f1
; CHECK: divsd
define double @f1() {
entry:
  %div = call double @llvm.experimental.constrained.fdiv.f64(
                                               double 1.000000e+00,
                                               double 1.000000e+01,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")
  ret double %div
}

; Verify that 'a - 0' isn't simplified to 'a' when the rounding mode is unknown.
;
; double f2(double a) {
;   // Because the result of '0 - 0' is negative zero if rounding mode is
;   // downward, this shouldn't be simplified.
;   return a - 0;
; }
;
; CHECK-LABEL: f2
; CHECK:  subsd
define double @f2(double %a) {
entry:
  %div = call double @llvm.experimental.constrained.fsub.f64(
                                               double %a,
                                               double 0.000000e+00,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")
  ret double %div
}

; Verify that '-((-a)*b)' isn't simplified to 'a*b' when the rounding mode is
; unknown.
;
; double f3(double a, double b) {
;   // Because the intermediate value involved in this calculation may require
;   // rounding, this shouldn't be simplified.
;   return -((-a)*b);
; }
;
; CHECK-LABEL: f3:
; CHECK:  subsd
; CHECK:  mulsd
; CHECK:  subsd
define double @f3(double %a, double %b) {
entry:
  %sub = call double @llvm.experimental.constrained.fsub.f64(
                                               double -0.000000e+00, double %a,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")
  %mul = call double @llvm.experimental.constrained.fmul.f64(
                                               double %sub, double %b,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")
  %ret = call double @llvm.experimental.constrained.fsub.f64(
                                               double -0.000000e+00,
                                               double %mul,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")
  ret double %ret
}

; Verify that FP operations are not performed speculatively when FP exceptions
; are not being ignored.
;
; double f4(int n, double a) {
;   // Because a + 1 may overflow, this should not be simplified.
;   if (n > 0)
;     return a + 1.0;
;   return a;
; }
;
; 
; CHECK-LABEL: f4:
; CHECK: testl
; CHECK: jle
; CHECK: addsd
define double @f4(i32 %n, double %a) {
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %add = call double @llvm.experimental.constrained.fadd.f64(
                                               double 1.000000e+00, double %a,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")
  br label %if.end

if.end:
  %a.0 = phi double [%add, %if.then], [ %a, %entry ]
  ret double %a.0
}


@llvm.fp.env = thread_local global i8 zeroinitializer, section "llvm.metadata"
declare double @llvm.experimental.constrained.fdiv.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.fmul.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.fadd.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.fsub.f64(double, double, metadata, metadata)

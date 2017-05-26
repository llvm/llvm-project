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

; Verify that sqrt(42.0) isn't simplified when the rounding mode is unknown.
; CHECK-LABEL: f5
; CHECK:  sqrtsd
define double @f5() {
entry:
  %result = call double @llvm.experimental.constrained.sqrt.f64(double 42.0,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")
  ret double %result
}

; Verify that pow(42.1, 3.0) isn't simplified when the rounding mode is unknown.
; CHECK-LABEL: f6
; CHECK:  pow
define double @f6() {
entry:
  %result = call double @llvm.experimental.constrained.pow.f64(double 42.1,
                                               double 3.0,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")
  ret double %result
}

; Verify that powi(42.1, 3) isn't simplified when the rounding mode is unknown.
; CHECK-LABEL: f7
; CHECK:  powi
define double @f7() {
entry:
  %result = call double @llvm.experimental.constrained.powi.f64(double 42.1,
                                               i32 3,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")
  ret double %result
}

; Verify that sin(42.0) isn't simplified when the rounding mode is unknown.
; CHECK-LABEL: f8
; CHECK:  sin
define double @f8() {
entry:
  %result = call double @llvm.experimental.constrained.sin.f64(double 42.0,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")
  ret double %result
}

; Verify that cos(42.0) isn't simplified when the rounding mode is unknown.
; CHECK-LABEL: f9
; CHECK:  cos
define double @f9() {
entry:
  %result = call double @llvm.experimental.constrained.cos.f64(double 42.0,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")
  ret double %result
}

; Verify that exp(42.0) isn't simplified when the rounding mode is unknown.
; CHECK-LABEL: f10
; CHECK:  exp
define double @f10() {
entry:
  %result = call double @llvm.experimental.constrained.exp.f64(double 42.0,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")
  ret double %result
}

; Verify that exp2(42.1) isn't simplified when the rounding mode is unknown.
; CHECK-LABEL: f11
; CHECK:  exp2
define double @f11() {
entry:
  %result = call double @llvm.experimental.constrained.exp2.f64(double 42.1,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")
  ret double %result
}

; Verify that log(42.0) isn't simplified when the rounding mode is unknown.
; CHECK-LABEL: f12
; CHECK:  log
define double @f12() {
entry:
  %result = call double @llvm.experimental.constrained.log.f64(double 42.0,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")
  ret double %result
}

; Verify that log10(42.0) isn't simplified when the rounding mode is unknown.
; CHECK-LABEL: f13
; CHECK:  log10
define double @f13() {
entry:
  %result = call double @llvm.experimental.constrained.log10.f64(double 42.0,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")
  ret double %result
}

; Verify that log2(42.0) isn't simplified when the rounding mode is unknown.
; CHECK-LABEL: f14
; CHECK:  log2
define double @f14() {
entry:
  %result = call double @llvm.experimental.constrained.log2.f64(double 42.0,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")
  ret double %result
}

; Verify that rint(42.1) isn't simplified when the rounding mode is unknown.
; CHECK-LABEL: f15
; CHECK:  rint
define double @f15() {
entry:
  %result = call double @llvm.experimental.constrained.rint.f64(double 42.1,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")
  ret double %result
}

; Verify that nearbyint(42.1) isn't simplified when the rounding mode is
; unknown.
; CHECK-LABEL: f16
; CHECK:  nearbyint
define double @f16() {
entry:
  %result = call double @llvm.experimental.constrained.nearbyint.f64(
                                               double 42.1,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")
  ret double %result
}

@llvm.fp.env = thread_local global i8 zeroinitializer, section "llvm.metadata"
declare double @llvm.experimental.constrained.fdiv.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.fmul.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.fadd.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.fsub.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.sqrt.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.pow.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.powi.f64(double, i32, metadata, metadata)
declare double @llvm.experimental.constrained.sin.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.cos.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.exp.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.exp2.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.log.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.log10.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.log2.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.rint.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.nearbyint.f64(double, metadata, metadata)

; RUN: opt -O3 -S < %s | FileCheck %s

; Test to verify that constants aren't folded when the rounding mode is unknown.
; CHECK-LABEL: @f1
; CHECK: call double @llvm.experimental.constrained.fdiv.f64
define double @f1() #0 {
entry:
  %div = call double @llvm.experimental.constrained.fdiv.f64(
                                               double 1.000000e+00,
                                               double 1.000000e+01,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret double %div
}

; Verify that 'a - 0' isn't simplified to 'a' when the rounding mode is unknown.
;
; double f2(double a) {
;   // Because the result of '0 - 0' is negative zero if rounding mode is
;   // downward, this shouldn't be simplified.
;   return a - 0.0;
; }
;
; CHECK-LABEL: @f2
; CHECK: call double @llvm.experimental.constrained.fsub.f64
define double @f2(double %a) #0 {
entry:
  %div = call double @llvm.experimental.constrained.fsub.f64(
                                               double %a, double 0.000000e+00,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
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
; CHECK-LABEL: @f3
; CHECK: call double @llvm.experimental.constrained.fsub.f64
; CHECK: call double @llvm.experimental.constrained.fmul.f64
; CHECK: call double @llvm.experimental.constrained.fsub.f64
define double @f3(double %a, double %b) #0 {
entry:
  %sub = call double @llvm.experimental.constrained.fsub.f64(
                                               double -0.000000e+00, double %a,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  %mul = call double @llvm.experimental.constrained.fmul.f64(
                                               double %sub, double %b,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  %ret = call double @llvm.experimental.constrained.fsub.f64(
                                               double -0.000000e+00,
                                               double %mul,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
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
; CHECK-LABEL: @f4
; CHECK-NOT: select
; CHECK: br i1 %cmp
define double @f4(i32 %n, double %a) #0 {
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %add = call double @llvm.experimental.constrained.fadd.f64(
                                               double 1.000000e+00, double %a,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  br label %if.end

if.end:
  %a.0 = phi double [%add, %if.then], [ %a, %entry ]
  ret double %a.0
}

; Verify that sqrt(42.0) isn't simplified when the rounding mode is unknown.
; CHECK-LABEL: f5
; CHECK: call double @llvm.experimental.constrained.sqrt
define double @f5() #0 {
entry:
  %result = call double @llvm.experimental.constrained.sqrt.f64(double 42.0,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret double %result
}

; Verify that pow(42.1, 3.0) isn't simplified when the rounding mode is unknown.
; CHECK-LABEL: f6
; CHECK: call double @llvm.experimental.constrained.pow
define double @f6() #0 {
entry:
  %result = call double @llvm.experimental.constrained.pow.f64(double 42.1,
                                               double 3.0,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret double %result
}

; Verify that powi(42.1, 3) isn't simplified when the rounding mode is unknown.
; CHECK-LABEL: f7
; CHECK: call double @llvm.experimental.constrained.powi
define double @f7() #0 {
entry:
  %result = call double @llvm.experimental.constrained.powi.f64(double 42.1,
                                               i32 3,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret double %result
}

; Verify that sin(42.0) isn't simplified when the rounding mode is unknown.
; CHECK-LABEL: f8
; CHECK: call double @llvm.experimental.constrained.sin
define double @f8() #0 {
entry:
  %result = call double @llvm.experimental.constrained.sin.f64(double 42.0,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret double %result
}

; Verify that cos(42.0) isn't simplified when the rounding mode is unknown.
; CHECK-LABEL: f9
; CHECK: call double @llvm.experimental.constrained.cos
define double @f9() #0 {
entry:
  %result = call double @llvm.experimental.constrained.cos.f64(double 42.0,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret double %result
}

; Verify that tan(42.0) isn't simplified when the rounding mode is unknown.
; CHECK-LABEL: ftan
; CHECK: call double @llvm.experimental.constrained.tan
define double @ftan() #0 {
entry:
  %result = call double @llvm.experimental.constrained.tan.f64(double 42.0,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret double %result
}

; Verify that exp(42.0) isn't simplified when the rounding mode is unknown.
; CHECK-LABEL: f10
; CHECK: call double @llvm.experimental.constrained.exp
define double @f10() #0 {
entry:
  %result = call double @llvm.experimental.constrained.exp.f64(double 42.0,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret double %result
}

; Verify that exp2(42.1) isn't simplified when the rounding mode is unknown.
; CHECK-LABEL: f11
; CHECK: call double @llvm.experimental.constrained.exp2
define double @f11() #0 {
entry:
  %result = call double @llvm.experimental.constrained.exp2.f64(double 42.1,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret double %result
}

; Verify that log(42.0) isn't simplified when the rounding mode is unknown.
; CHECK-LABEL: f12
; CHECK: call double @llvm.experimental.constrained.log
define double @f12() #0 {
entry:
  %result = call double @llvm.experimental.constrained.log.f64(double 42.0,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret double %result
}

; Verify that log10(42.0) isn't simplified when the rounding mode is unknown.
; CHECK-LABEL: f13
; CHECK: call double @llvm.experimental.constrained.log10
define double @f13() #0 {
entry:
  %result = call double @llvm.experimental.constrained.log10.f64(double 42.0,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret double %result
}

; Verify that log2(42.0) isn't simplified when the rounding mode is unknown.
; CHECK-LABEL: f14
; CHECK: call double @llvm.experimental.constrained.log2
define double @f14() #0 {
entry:
  %result = call double @llvm.experimental.constrained.log2.f64(double 42.0,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret double %result
}

; Verify that rint(42.1) isn't simplified when the rounding mode is unknown.
; CHECK-LABEL: f15
; CHECK: call double @llvm.experimental.constrained.rint
define double @f15() #0 {
entry:
  %result = call double @llvm.experimental.constrained.rint.f64(double 42.1,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret double %result
}

; Verify that nearbyint(42.1) isn't simplified when the rounding mode is
; unknown.
; CHECK-LABEL: f16
; CHECK: call double @llvm.experimental.constrained.nearbyint
define double @f16() #0 {
entry:
  %result = call double @llvm.experimental.constrained.nearbyint.f64(
                                               double 42.1,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret double %result
}

; Verify that fma(42.1) isn't simplified when the rounding mode is
; unknown.
; CHECK-LABEL: f17
; CHECK: call double @llvm.experimental.constrained.fma
define double @f17() #0 {
entry:
  %result = call double @llvm.experimental.constrained.fma.f64(double 42.1, double 42.1, double 42.1,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret double %result
}

; Verify that fptoui(42.1) isn't simplified when the rounding mode is
; unknown.
; CHECK-LABEL: f18
; CHECK: call zeroext i32 @llvm.experimental.constrained.fptoui
define zeroext i32 @f18() #0 {
entry:
  %result = call zeroext i32 @llvm.experimental.constrained.fptoui.i32.f64(
                                               double 42.1,
                                               metadata !"fpexcept.strict") #0
  ret i32 %result
}

; Verify that fptosi(42.1) isn't simplified when the rounding mode is
; unknown.
; CHECK-LABEL: f19
; CHECK: call i32 @llvm.experimental.constrained.fptosi
define i32 @f19() #0 {
entry:
  %result = call i32 @llvm.experimental.constrained.fptosi.i32.f64(double 42.1,
                                               metadata !"fpexcept.strict") #0
  ret i32 %result
}

; Verify that fptrunc(42.1) isn't simplified when the rounding mode is
; unknown.
; CHECK-LABEL: f20
; CHECK: call float @llvm.experimental.constrained.fptrunc
define float @f20() #0 {
entry:
  %result = call float @llvm.experimental.constrained.fptrunc.f32.f64(
                                               double 42.1,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret float %result
}

; Verify that fpext(42.1) isn't simplified when the rounding mode is
; unknown.
; CHECK-LABEL: f21
; CHECK: call double @llvm.experimental.constrained.fpext
define double @f21() #0 {
entry:
  %result = call double @llvm.experimental.constrained.fpext.f64.f32(float 42.0,
                                               metadata !"fpexcept.strict") #0
  ret double %result
}

; Verify that lrint(42.1) isn't simplified when the rounding mode is unknown.
; CHECK-LABEL: f22
; CHECK: call i32 @llvm.experimental.constrained.lrint
define i32 @f22() #0 {
entry:
  %result = call i32 @llvm.experimental.constrained.lrint.i32.f64(double 42.1,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret i32 %result
}

; Verify that lrintf(42.0) isn't simplified when the rounding mode is unknown.
; CHECK-LABEL: f23
; CHECK: call i32 @llvm.experimental.constrained.lrint
define i32 @f23() #0 {
entry:
  %result = call i32 @llvm.experimental.constrained.lrint.i32.f32(float 42.0,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret i32 %result
}

; Verify that llrint(42.1) isn't simplified when the rounding mode is unknown.
; CHECK-LABEL: f24
; CHECK: call i64 @llvm.experimental.constrained.llrint
define i64 @f24() #0 {
entry:
  %result = call i64 @llvm.experimental.constrained.llrint.i64.f64(double 42.1,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret i64 %result
}

; Verify that llrint(42.0) isn't simplified when the rounding mode is unknown.
; CHECK-LABEL: f25
; CHECK: call i64 @llvm.experimental.constrained.llrint
define i64 @f25() #0 {
entry:
  %result = call i64 @llvm.experimental.constrained.llrint.i64.f32(float 42.0,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret i64 %result
}

; Verify that lround(42.1) isn't simplified when the rounding mode is unknown.
; CHECK-LABEL: f26
; CHECK: call i32 @llvm.experimental.constrained.lround
define i32 @f26() #0 {
entry:
  %result = call i32 @llvm.experimental.constrained.lround.i32.f64(double 42.1,
                                               metadata !"fpexcept.strict") #0
  ret i32 %result
}

; Verify that lround(42.0) isn't simplified when the rounding mode is unknown.
; CHECK-LABEL: f27
; CHECK: call i32 @llvm.experimental.constrained.lround
define i32 @f27() #0 {
entry:
  %result = call i32 @llvm.experimental.constrained.lround.i32.f32(float 42.0,
                                               metadata !"fpexcept.strict") #0
  ret i32 %result
}

; Verify that llround(42.1) isn't simplified when the rounding mode is unknown.
; CHECK-LABEL: f28
; CHECK: call i64 @llvm.experimental.constrained.llround
define i64 @f28() #0 {
entry:
  %result = call i64 @llvm.experimental.constrained.llround.i64.f64(double 42.1,
                                               metadata !"fpexcept.strict") #0
  ret i64 %result
}

; Verify that llround(42.0) isn't simplified when the rounding mode is unknown.
; CHECK-LABEL: f29
; CHECK: call i64 @llvm.experimental.constrained.llround
define i64 @f29() #0 {
entry:
  %result = call i64 @llvm.experimental.constrained.llround.i64.f32(float 42.0,
                                               metadata !"fpexcept.strict") #0
  ret i64 %result
}

; Verify that sitofp(42) isn't simplified when the rounding mode is unknown.
; CHECK-LABEL: @f30
; CHECK: call double @llvm.experimental.constrained.sitofp
define double @f30() #0 {
entry:
  %result = call double @llvm.experimental.constrained.sitofp.f64.i32(i32 42,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret double %result
}

; Verify that uitofp(42) isn't simplified when the rounding mode is unknown.
; CHECK-LABEL: @f31
; CHECK: call double @llvm.experimental.constrained.uitofp
define double @f31() #0 {
entry:
  %result = call double @llvm.experimental.constrained.uitofp.f64.i32(i32 42,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret double %result
}

attributes #0 = { strictfp }

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
declare double @llvm.experimental.constrained.tan.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.exp.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.exp2.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.log.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.log10.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.log2.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.rint.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.nearbyint.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.fma.f64(double, double, double, metadata, metadata)
declare i32 @llvm.experimental.constrained.fptosi.i32.f64(double, metadata)
declare i32 @llvm.experimental.constrained.fptoui.i32.f64(double, metadata)
declare float @llvm.experimental.constrained.fptrunc.f32.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.fpext.f64.f32(float, metadata)
declare i32 @llvm.experimental.constrained.lrint.i32.f64(double, metadata, metadata)
declare i32 @llvm.experimental.constrained.lrint.i32.f32(float, metadata, metadata)
declare i64 @llvm.experimental.constrained.llrint.i64.f64(double, metadata, metadata)
declare i64 @llvm.experimental.constrained.llrint.i64.f32(float, metadata, metadata)
declare i32 @llvm.experimental.constrained.lround.i32.f64(double, metadata)
declare i32 @llvm.experimental.constrained.lround.i32.f32(float, metadata)
declare i64 @llvm.experimental.constrained.llround.i64.f64(double, metadata)
declare i64 @llvm.experimental.constrained.llround.i64.f32(float, metadata)
declare double @llvm.experimental.constrained.sitofp.f64.i32(i32, metadata, metadata)
declare double @llvm.experimental.constrained.uitofp.f64.i32(i32, metadata, metadata)

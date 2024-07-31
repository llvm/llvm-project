; RUN: opt < %s -passes=libcalls-shrinkwrap -S | FileCheck %s

; #include <math.h>
; #include <fenv.h>
; #include <stdlib.h>
;
; void() {
;   volatile double d;
;   d = __builtin_nan ("");
;   feclearexcept (FE_ALL_EXCEPT);
;   acos(d);
;   if (fetestexcept (FE_ALL_EXCEPT))  // expect no fp exception raised
;     abort();
; }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @test_quiet_nan() {
  %1 = alloca double, align 8
  store volatile double 0x7FF8000000000000, ptr %1, align 8
  %2 = tail call i32 @feclearexcept(i32 noundef 61)
  %3 = load volatile double, ptr %1, align 8
  %4 = call double @acos(double noundef %3)
; CHECK: [[COND1:%[0-9]+]] = fcmp ogt double [[VALUE:%.*]], 1.000000e+00
; CHECK: [[COND1:%[0-9]+]] = fcmp olt double [[VALUE]], -1.000000e+00
  %5 = call i32 @fetestexcept(i32 noundef 61)
  %6 = icmp ne i32 %5, 0
  br i1 %6, label %abort, label %ret

abort:
  call void @abort()
  unreachable

ret:
  ret void
}

define void @test_quiet_nan_strictfp() strictfp {
  %1 = alloca double, align 8
  store volatile double 0x7FF8000000000000, ptr %1, align 8
  %2 = tail call i32 @feclearexcept(i32 noundef 61) strictfp
  %3 = load volatile double, ptr %1, align 8
  %4 = call double @acos(double noundef %3) strictfp
; Generate constrained fcmp if function has strictfp attribute.
; That avoids raising fp exception with quiet nan input.
; CHECK: [[COND1:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f64(double [[VALUE]], double 1.000000e+00, metadata !"ogt", metadata !"fpexcept.strict")
; CHECK: [[COND1:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f64(double [[VALUE]], double -1.000000e+00, metadata !"olt", metadata !"fpexcept.strict")
  %5 = call i32 @fetestexcept(i32 noundef 61) strictfp
  %6 = icmp ne i32 %5, 0
  br i1 %6, label %abort, label %ret

abort:
  call void @abort() strictfp
  unreachable

ret:
  ret void
}

declare i32 @feclearexcept(i32 noundef)

declare i32 @fetestexcept(i32 noundef)

declare double @acos(double noundef)

declare void @abort()

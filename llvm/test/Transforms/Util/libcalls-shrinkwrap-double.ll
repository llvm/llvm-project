; RUN: opt < %s -passes=libcalls-shrinkwrap -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @test_range_error(double %value) {
entry:
  %call_0 = call double @cosh(double %value)
; CHECK: [[COND1:%[0-9]+]] = fcmp olt double %value, -7.100000e+02
; CHECK: [[COND2:%[0-9]+]] = fcmp ogt double %value, 7.100000e+02
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT:[0-9]+]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_0 = call double @cosh(double %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_1 = call double @exp(double %value)
; CHECK: [[COND1:%[0-9]+]] = fcmp olt double %value, -7.450000e+02
; CHECK: [[COND2:%[0-9]+]] = fcmp ogt double %value, 7.090000e+02
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_1 = call double @exp(double %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_3 = call double @exp2(double %value)
; CHECK: [[COND1:%[0-9]+]] = fcmp olt double %value, -1.074000e+03
; CHECK: [[COND2:%[0-9]+]] = fcmp ogt double %value, 1.023000e+03
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_3 = call double @exp2(double %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_4 = call double @sinh(double %value)
; CHECK: [[COND1:%[0-9]+]] = fcmp olt double %value, -7.100000e+02
; CHECK: [[COND2:%[0-9]+]] = fcmp ogt double %value, 7.100000e+02
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_4 = call double @sinh(double %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_5 = call double @expm1(double %value)
; CHECK: [[COND:%[0-9]+]] = fcmp ogt double %value, 7.090000e+02
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_5 = call double @expm1(double %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  ret void
}

define void @test_range_error_strictfp(double %value) strictfp {
entry:
  %call_0 = call double @cosh(double %value)
; CHECK: [[COND1:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %value, double -7.100000e+02, metadata !"olt", metadata !"fpexcept.strict") #0
; CHECK: [[COND2:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %value, double 7.100000e+02, metadata !"ogt", metadata !"fpexcept.strict") #0
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT:[0-9]+]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_0 = call double @cosh(double %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_1 = call double @exp(double %value)
; CHECK: [[COND1:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %value, double -7.450000e+02, metadata !"olt", metadata !"fpexcept.strict") #0
; CHECK: [[COND2:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %value, double 7.090000e+02, metadata !"ogt", metadata !"fpexcept.strict") #0
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_1 = call double @exp(double %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_3 = call double @exp2(double %value)
; CHECK: [[COND1:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %value, double -1.074000e+03, metadata !"olt", metadata !"fpexcept.strict") #0
; CHECK: [[COND2:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %value, double 1.023000e+03, metadata !"ogt", metadata !"fpexcept.strict") #0
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_3 = call double @exp2(double %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_4 = call double @sinh(double %value)
; CHECK: [[COND1:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %value, double -7.100000e+02, metadata !"olt", metadata !"fpexcept.strict") #0
; CHECK: [[COND2:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %value, double 7.100000e+02, metadata !"ogt", metadata !"fpexcept.strict") #0
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_4 = call double @sinh(double %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_5 = call double @expm1(double %value)
; CHECK: [[COND:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %value, double 7.090000e+02, metadata !"ogt", metadata !"fpexcept.strict") #0
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_5 = call double @expm1(double %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  ret void
}

declare double @cosh(double)
declare double @exp(double)
declare double @exp2(double)
declare double @sinh(double)
declare double @expm1(double)

define void @test_domain_error(double %value) {
entry:
  %call_00 = call double @acos(double %value)
; CHECK: [[COND1:%[0-9]+]] = fcmp ogt double %value, 1.000000e+00
; CHECK: [[COND2:%[0-9]+]] = fcmp olt double %value, -1.000000e+00
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_00 = call double @acos(double %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_01 = call double @asin(double %value)
; CHECK: [[COND1:%[0-9]+]] = fcmp ogt double %value, 1.000000e+00
; CHECK: [[COND2:%[0-9]+]] = fcmp olt double %value, -1.000000e+00
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_01 = call double @asin(double %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_02 = call double @cos(double %value)
; CHECK: [[COND1:%[0-9]+]] = fcmp oeq double %value, 0xFFF0000000000000
; CHECK: [[COND2:%[0-9]+]] = fcmp oeq double %value, 0x7FF0000000000000
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_02 = call double @cos(double %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_03 = call double @sin(double %value)
; CHECK: [[COND1:%[0-9]+]] = fcmp oeq double %value, 0xFFF0000000000000
; CHECK: [[COND2:%[0-9]+]] = fcmp oeq double %value, 0x7FF0000000000000
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_03 = call double @sin(double %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_04 = call double @acosh(double %value)
; CHECK: [[COND:%[0-9]+]] = fcmp olt double %value, 1.000000e+00
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_04 = call double @acosh(double %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_05 = call double @sqrt(double %value)
; CHECK: [[COND:%[0-9]+]] = fcmp olt double %value, 0.000000e+00
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_05 = call double @sqrt(double %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_06 = call double @atanh(double %value)
; CHECK: [[COND1:%[0-9]+]] = fcmp oge double %value, 1.000000e+00
; CHECK: [[COND2:%[0-9]+]] = fcmp ole double %value, -1.000000e+00
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_06 = call double @atanh(double %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_07 = call double @log(double %value)
; CHECK: [[COND:%[0-9]+]] = fcmp ole double %value, 0.000000e+00
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_07 = call double @log(double %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_08 = call double @log10(double %value)
; CHECK: [[COND:%[0-9]+]] = fcmp ole double %value, 0.000000e+00
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_08 = call double @log10(double %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_09 = call double @log2(double %value)
; CHECK: [[COND:%[0-9]+]] = fcmp ole double %value, 0.000000e+00
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_09 = call double @log2(double %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_10 = call double @logb(double %value)
; CHECK: [[COND:%[0-9]+]] = fcmp ole double %value, 0.000000e+00
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_10 = call double @logb(double %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_11 = call double @log1p(double %value)
; CHECK: [[COND:%[0-9]+]] = fcmp ole double %value, -1.000000e+00
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_11 = call double @log1p(double %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  ret void
}

define void @test_domain_error_strictfp(double %value) strictfp {
entry:
  %call_00 = call double @acos(double %value)
; CHECK: [[COND1:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %value, double 1.000000e+00, metadata !"ogt", metadata !"fpexcept.strict") #0
; CHECK: [[COND2:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %value, double -1.000000e+00, metadata !"olt", metadata !"fpexcept.strict") #0
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_00 = call double @acos(double %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_01 = call double @asin(double %value)
; CHECK: [[COND1:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %value, double 1.000000e+00, metadata !"ogt", metadata !"fpexcept.strict") #0
; CHECK: [[COND2:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %value, double -1.000000e+00, metadata !"olt", metadata !"fpexcept.strict") #0
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_01 = call double @asin(double %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_02 = call double @cos(double %value)
; CHECK: [[COND1:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %value, double 0xFFF0000000000000, metadata !"oeq", metadata !"fpexcept.strict") #0
; CHECK: [[COND2:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %value, double 0x7FF0000000000000, metadata !"oeq", metadata !"fpexcept.strict") #0
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_02 = call double @cos(double %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_03 = call double @sin(double %value)
; CHECK: [[COND1:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %value, double 0xFFF0000000000000, metadata !"oeq", metadata !"fpexcept.strict") #0
; CHECK: [[COND2:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %value, double 0x7FF0000000000000, metadata !"oeq", metadata !"fpexcept.strict") #0
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_03 = call double @sin(double %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_04 = call double @acosh(double %value)
; CHECK: [[COND:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %value, double 1.000000e+00, metadata !"olt", metadata !"fpexcept.strict") #0
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_04 = call double @acosh(double %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_05 = call double @sqrt(double %value)
; CHECK: [[COND:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %value, double 0.000000e+00, metadata !"olt", metadata !"fpexcept.strict") #0
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_05 = call double @sqrt(double %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_06 = call double @atanh(double %value)
; CHECK: [[COND1:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %value, double 1.000000e+00, metadata !"oge", metadata !"fpexcept.strict") #0
; CHECK: [[COND2:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %value, double -1.000000e+00, metadata !"ole", metadata !"fpexcept.strict") #0
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_06 = call double @atanh(double %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_07 = call double @log(double %value)
; CHECK: [[COND:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %value, double 0.000000e+00, metadata !"ole", metadata !"fpexcept.strict") #0
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_07 = call double @log(double %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_08 = call double @log10(double %value)
; CHECK: [[COND:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %value, double 0.000000e+00, metadata !"ole", metadata !"fpexcept.strict") #0
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_08 = call double @log10(double %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_09 = call double @log2(double %value)
; CHECK: [[COND:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %value, double 0.000000e+00, metadata !"ole", metadata !"fpexcept.strict") #0
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_09 = call double @log2(double %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_10 = call double @logb(double %value)
; CHECK: [[COND:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %value, double 0.000000e+00, metadata !"ole", metadata !"fpexcept.strict") #0
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_10 = call double @logb(double %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_11 = call double @log1p(double %value)
; CHECK: [[COND:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %value, double -1.000000e+00, metadata !"ole", metadata !"fpexcept.strict") #0
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_11 = call double @log1p(double %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  ret void
}

declare double @acos(double)
declare double @asin(double)
declare double @cos(double)
declare double @sin(double)
declare double @acosh(double)
declare double @sqrt(double)
declare double @atanh(double)
declare double @log(double)
declare double @log10(double)
declare double @log2(double)
declare double @logb(double)
declare double @log1p(double)

define void @test_pow(i32 %int_val, double %exp) {
  %call = call double @pow(double 2.500000e+00, double %exp)
; CHECK: [[COND:%[0-9]+]] = fcmp ogt double %exp, 1.270000e+02
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call = call double @pow(double 2.500000e+00, double %exp)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %conv = sitofp i32 %int_val to double
  %call1 = call double @pow(double %conv, double %exp)
; CHECK: [[COND1:%[0-9]+]] = fcmp ogt double %exp, 3.200000e+01
; CHECK: [[COND2:%[0-9]+]] = fcmp ole double %conv, 0.000000e+00
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call1 = call double @pow(double %conv, double %exp)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %conv2 = trunc i32 %int_val to i8
  %conv3 = uitofp i8 %conv2 to double
  %call4 = call double @pow(double %conv3, double %exp)
; CHECK: [[COND1:%[0-9]+]] = fcmp ogt double %exp, 1.280000e+02
; CHECK: [[COND2:%[0-9]+]] = fcmp ole double %conv3, 0.000000e+00
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call4 = call double @pow(double %conv3, double %exp)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:


  %conv5 = trunc i32 %int_val to i16
  %conv6 = uitofp i16 %conv5 to double
  %call7 = call double @pow(double %conv6, double %exp)
; CHECK: [[COND1:%[0-9]+]] = fcmp ogt double %exp, 6.400000e+01
; CHECK: [[COND2:%[0-9]+]] = fcmp ole double %conv6, 0.000000e+00
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call7 = call double @pow(double %conv6, double %exp)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  ret void
}

define void @test_pow_strictfp(i32 %int_val, double %exp) strictfp {
  %call = call double @pow(double 2.500000e+00, double %exp)
; CHECK: [[COND:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %exp, double 1.270000e+02, metadata !"ogt", metadata !"fpexcept.strict") #0
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call = call double @pow(double 2.500000e+00, double %exp)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %conv = sitofp i32 %int_val to double
  %call1 = call double @pow(double %conv, double %exp)
; CHECK: [[COND1:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %exp, double 3.200000e+01, metadata !"ogt", metadata !"fpexcept.strict") #0
; CHECK: [[COND2:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %conv, double 0.000000e+00, metadata !"ole", metadata !"fpexcept.strict") #0
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call1 = call double @pow(double %conv, double %exp)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %conv2 = trunc i32 %int_val to i8
  %conv3 = uitofp i8 %conv2 to double
  %call4 = call double @pow(double %conv3, double %exp)
; CHECK: [[COND1:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %exp, double 1.280000e+02, metadata !"ogt", metadata !"fpexcept.strict") #0
; CHECK: [[COND2:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %conv3, double 0.000000e+00, metadata !"ole", metadata !"fpexcept.strict") #0
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call4 = call double @pow(double %conv3, double %exp)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:


  %conv5 = trunc i32 %int_val to i16
  %conv6 = uitofp i16 %conv5 to double
  %call7 = call double @pow(double %conv6, double %exp)
; CHECK: [[COND1:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %exp, double 6.400000e+01, metadata !"ogt", metadata !"fpexcept.strict") #0
; CHECK: [[COND2:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %conv6, double 0.000000e+00, metadata !"ole", metadata !"fpexcept.strict") #0
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call7 = call double @pow(double %conv6, double %exp)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  ret void
}

declare double @pow(double, double)

; CHECK: ![[BRANCH_WEIGHT]] = !{!"branch_weights", i32 1, i32 2000}

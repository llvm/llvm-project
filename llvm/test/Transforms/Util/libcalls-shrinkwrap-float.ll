; RUN: opt < %s -passes=libcalls-shrinkwrap -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @test_range_error(float %value) {
entry:
  %call_0 = call float @coshf(float %value)
; CHECK: [[COND1:%[0-9]+]] = fcmp olt float %value, -8.900000e+01
; CHECK: [[COND2:%[0-9]+]] = fcmp ogt float %value, 8.900000e+01
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT:[0-9]+]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_0 = call float @coshf(float %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_1 = call float @expf(float %value)
; CHECK: [[COND1:%[0-9]+]] = fcmp olt float %value, -1.030000e+02
; CHECK: [[COND2:%[0-9]+]] = fcmp ogt float %value, 8.800000e+01
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_1 = call float @expf(float %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_3 = call float @exp2f(float %value)
; CHECK: [[COND1:%[0-9]+]] = fcmp olt float %value, -1.490000e+02
; CHECK: [[COND2:%[0-9]+]] = fcmp ogt float %value, 1.270000e+02
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_3 = call float @exp2f(float %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_4 = call float @sinhf(float %value)
; CHECK: [[COND1:%[0-9]+]] = fcmp olt float %value, -8.900000e+01
; CHECK: [[COND2:%[0-9]+]] = fcmp ogt float %value, 8.900000e+01
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_4 = call float @sinhf(float %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_5 = call float @expm1f(float %value)
; CHECK: [[COND:%[0-9]+]] = fcmp ogt float %value, 8.800000e+01
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_5 = call float @expm1f(float %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  ret void
}

define void @test_range_error_strictfp(float %value) strictfp {
entry:
  %call_0 = call float @coshf(float %value) strictfp
; CHECK: [[COND1:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f32(float [[VALUE:%.*]], float -8.900000e+01, metadata !"olt", metadata !"fpexcept.strict")
; CHECK: [[COND2:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f32(float [[VALUE]], float 8.900000e+01, metadata !"ogt", metadata !"fpexcept.strict")
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT:[0-9]+]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_0 = call float @coshf(float %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_1 = call float @expf(float %value) strictfp
; CHECK: [[COND1:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f32(float [[VALUE]], float -1.030000e+02, metadata !"olt", metadata !"fpexcept.strict")
; CHECK: [[COND2:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f32(float [[VALUE]], float 8.800000e+01, metadata !"ogt", metadata !"fpexcept.strict")
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_1 = call float @expf(float %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_3 = call float @exp2f(float %value) strictfp
; CHECK: [[COND1:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f32(float [[VALUE]], float -1.490000e+02, metadata !"olt", metadata !"fpexcept.strict")
; CHECK: [[COND2:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f32(float [[VALUE]], float 1.270000e+02, metadata !"ogt", metadata !"fpexcept.strict")
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_3 = call float @exp2f(float %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_4 = call float @sinhf(float %value) strictfp
; CHECK: [[COND1:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f32(float [[VALUE]], float -8.900000e+01, metadata !"olt", metadata !"fpexcept.strict")
; CHECK: [[COND2:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f32(float [[VALUE]], float 8.900000e+01, metadata !"ogt", metadata !"fpexcept.strict")
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_4 = call float @sinhf(float %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_5 = call float @expm1f(float %value) strictfp
; CHECK: [[COND:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f32(float [[VALUE]], float 8.800000e+01, metadata !"ogt", metadata !"fpexcept.strict")
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_5 = call float @expm1f(float %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  ret void
}

declare float @coshf(float)
declare float @expf(float)
declare float @exp2f(float)
declare float @sinhf(float)
declare float @expm1f(float)

define void @test_domain_error(float %value) {
entry:

  %call_00 = call float @acosf(float %value)
; CHECK: [[COND1:%[0-9]+]] = fcmp ogt float %value, 1.000000e+00
; CHECK: [[COND2:%[0-9]+]] = fcmp olt float %value, -1.000000e+00
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_00 = call float @acosf(float %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_01 = call float @asinf(float %value)
; CHECK: [[COND1:%[0-9]+]] = fcmp ogt float %value, 1.000000e+00
; CHECK: [[COND2:%[0-9]+]] = fcmp olt float %value, -1.000000e+00
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_01 = call float @asinf(float %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_02 = call float @cosf(float %value)
; CHECK: [[COND1:%[0-9]+]] = fcmp oeq float %value, 0xFFF0000000000000
; CHECK: [[COND2:%[0-9]+]] = fcmp oeq float %value, 0x7FF0000000000000
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_02 = call float @cosf(float %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_03 = call float @sinf(float %value)
; CHECK: [[COND1:%[0-9]+]] = fcmp oeq float %value, 0xFFF0000000000000
; CHECK: [[COND2:%[0-9]+]] = fcmp oeq float %value, 0x7FF0000000000000
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_03 = call float @sinf(float %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_04 = call float @acoshf(float %value)
; CHECK: [[COND:%[0-9]+]] = fcmp olt float %value, 1.000000e+00
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_04 = call float @acoshf(float %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_05 = call float @sqrtf(float %value)
; CHECK: [[COND:%[0-9]+]] = fcmp olt float %value, 0.000000e+00
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_05 = call float @sqrtf(float %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_06 = call float @atanhf(float %value)
; CHECK: [[COND1:%[0-9]+]] = fcmp oge float %value, 1.000000e+00
; CHECK: [[COND2:%[0-9]+]] = fcmp ole float %value, -1.000000e+00
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_06 = call float @atanhf(float %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_07 = call float @logf(float %value)
; CHECK: [[COND:%[0-9]+]] = fcmp ole float %value, 0.000000e+00
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_07 = call float @logf(float %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_08 = call float @log10f(float %value)
; CHECK: [[COND:%[0-9]+]] = fcmp ole float %value, 0.000000e+00
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_08 = call float @log10f(float %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_09 = call float @log2f(float %value)
; CHECK: [[COND:%[0-9]+]] = fcmp ole float %value, 0.000000e+00
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_09 = call float @log2f(float %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_10 = call float @logbf(float %value)
; CHECK: [[COND:%[0-9]+]] = fcmp ole float %value, 0.000000e+00
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_10 = call float @logbf(float %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_11 = call float @log1pf(float %value)
; CHECK: [[COND:%[0-9]+]] = fcmp ole float %value, -1.000000e+00
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_11 = call float @log1pf(float %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:
  ret void
}

define void @test_domain_error_strictfp(float %value) strictfp {
entry:

  %call_00 = call float @acosf(float %value) strictfp
; CHECK: [[COND1:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f32(float [[VALUE:%.*]], float 1.000000e+00, metadata !"ogt", metadata !"fpexcept.strict")
; CHECK: [[COND2:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f32(float [[VALUE]], float -1.000000e+00, metadata !"olt", metadata !"fpexcept.strict")
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_00 = call float @acosf(float %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_01 = call float @asinf(float %value) strictfp
; CHECK: [[COND1:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f32(float [[VALUE]], float 1.000000e+00, metadata !"ogt", metadata !"fpexcept.strict")
; CHECK: [[COND2:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f32(float [[VALUE]], float -1.000000e+00, metadata !"olt", metadata !"fpexcept.strict")
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_01 = call float @asinf(float %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_02 = call float @cosf(float %value) strictfp
; CHECK: [[COND1:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f32(float [[VALUE]], float 0xFFF0000000000000, metadata !"oeq", metadata !"fpexcept.strict")
; CHECK: [[COND2:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f32(float [[VALUE]], float 0x7FF0000000000000, metadata !"oeq", metadata !"fpexcept.strict")
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_02 = call float @cosf(float %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_03 = call float @sinf(float %value) strictfp
; CHECK: [[COND1:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f32(float [[VALUE]], float 0xFFF0000000000000, metadata !"oeq", metadata !"fpexcept.strict")
; CHECK: [[COND2:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f32(float [[VALUE]], float 0x7FF0000000000000, metadata !"oeq", metadata !"fpexcept.strict")
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_03 = call float @sinf(float %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_04 = call float @acoshf(float %value) strictfp
; CHECK: [[COND:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f32(float [[VALUE]], float 1.000000e+00, metadata !"olt", metadata !"fpexcept.strict")
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_04 = call float @acoshf(float %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_05 = call float @sqrtf(float %value) strictfp
; CHECK: [[COND:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f32(float [[VALUE]], float 0.000000e+00, metadata !"olt", metadata !"fpexcept.strict")
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_05 = call float @sqrtf(float %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_06 = call float @atanhf(float %value) strictfp
; CHECK: [[COND1:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f32(float [[VALUE]], float 1.000000e+00, metadata !"oge", metadata !"fpexcept.strict")
; CHECK: [[COND2:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f32(float [[VALUE]], float -1.000000e+00, metadata !"ole", metadata !"fpexcept.strict")
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_06 = call float @atanhf(float %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_07 = call float @logf(float %value) strictfp
; CHECK: [[COND:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f32(float [[VALUE]], float 0.000000e+00, metadata !"ole", metadata !"fpexcept.strict")
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_07 = call float @logf(float %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_08 = call float @log10f(float %value) strictfp
; CHECK: [[COND:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f32(float [[VALUE]], float 0.000000e+00, metadata !"ole", metadata !"fpexcept.strict")
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_08 = call float @log10f(float %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_09 = call float @log2f(float %value) strictfp
; CHECK: [[COND:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f32(float [[VALUE]], float 0.000000e+00, metadata !"ole", metadata !"fpexcept.strict")
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_09 = call float @log2f(float %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_10 = call float @logbf(float %value) strictfp
; CHECK: [[COND:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f32(float [[VALUE]], float 0.000000e+00, metadata !"ole", metadata !"fpexcept.strict")
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_10 = call float @logbf(float %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_11 = call float @log1pf(float %value) strictfp
; CHECK: [[COND:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f32(float [[VALUE]], float -1.000000e+00, metadata !"ole", metadata !"fpexcept.strict")
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_11 = call float @log1pf(float %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:
  ret void
}

declare float @acosf(float)
declare float @asinf(float)
declare float @cosf(float)
declare float @sinf(float)
declare float @acoshf(float)
declare float @sqrtf(float)
declare float @atanhf(float)
declare float @logf(float)
declare float @log10f(float)
declare float @log2f(float)
declare float @logbf(float)
declare float @log1pf(float)

; CHECK: ![[BRANCH_WEIGHT]] = !{!"branch_weights", i32 1, i32 2000}

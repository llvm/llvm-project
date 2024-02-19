; RUN: opt < %s -passes=libcalls-shrinkwrap -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @test_range_error(x86_fp80 %value) {
entry:
  %call_0 = call x86_fp80 @coshl(x86_fp80 %value)
; CHECK: [[COND1:%[0-9]+]] = fcmp olt x86_fp80 %value, 0xKC00CB174000000000000
; CHECK: [[COND2:%[0-9]+]] = fcmp ogt x86_fp80 %value, 0xK400CB174000000000000
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT:[0-9]+]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_0 = call x86_fp80 @coshl(x86_fp80 %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_1 = call x86_fp80 @expl(x86_fp80 %value)
; CHECK: [[COND1:%[0-9]+]] = fcmp olt x86_fp80 %value, 0xKC00CB21C000000000000
; CHECK: [[COND2:%[0-9]+]] = fcmp ogt x86_fp80 %value, 0xK400CB170000000000000
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_1 = call x86_fp80 @expl(x86_fp80 %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_3 = call x86_fp80 @exp2l(x86_fp80 %value)
; CHECK: [[COND1:%[0-9]+]] = fcmp olt x86_fp80 %value, 0xKC00D807A000000000000
; CHECK: [[COND2:%[0-9]+]] = fcmp ogt x86_fp80 %value, 0xK400CB1DC000000000000
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_3 = call x86_fp80 @exp2l(x86_fp80 %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_4 = call x86_fp80 @sinhl(x86_fp80 %value)
; CHECK: [[COND1:%[0-9]+]] = fcmp olt x86_fp80 %value, 0xKC00CB174000000000000
; CHECK: [[COND2:%[0-9]+]] = fcmp ogt x86_fp80 %value, 0xK400CB174000000000000
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_4 = call x86_fp80 @sinhl(x86_fp80 %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_5 = call x86_fp80 @expm1l(x86_fp80 %value)
; CHECK: [[COND:%[0-9]+]] = fcmp ogt x86_fp80 %value, 0xK400CB170000000000000
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_5 = call x86_fp80 @expm1l(x86_fp80 %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  ret void
}

define void @test_range_error_strictfp(x86_fp80 %value) strictfp {
entry:
  %call_0 = call x86_fp80 @coshl(x86_fp80 %value) strictfp
; CHECK: [[COND1:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f80(x86_fp80 [[VALUE:%.*]], x86_fp80 0xKC00CB174000000000000, metadata !"olt", metadata !"fpexcept.strict")
; CHECK: [[COND2:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f80(x86_fp80 [[VALUE]], x86_fp80 0xK400CB174000000000000, metadata !"ogt", metadata !"fpexcept.strict")
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT:[0-9]+]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_0 = call x86_fp80 @coshl(x86_fp80 %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_1 = call x86_fp80 @expl(x86_fp80 %value) strictfp
; CHECK: [[COND1:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f80(x86_fp80 [[VALUE]], x86_fp80 0xKC00CB21C000000000000, metadata !"olt", metadata !"fpexcept.strict")
; CHECK: [[COND2:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f80(x86_fp80 [[VALUE]], x86_fp80 0xK400CB170000000000000, metadata !"ogt", metadata !"fpexcept.strict")
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_1 = call x86_fp80 @expl(x86_fp80 %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_3 = call x86_fp80 @exp2l(x86_fp80 %value) strictfp
; CHECK: [[COND1:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f80(x86_fp80 [[VALUE]], x86_fp80 0xKC00D807A000000000000, metadata !"olt", metadata !"fpexcept.strict")
; CHECK: [[COND2:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f80(x86_fp80 [[VALUE]], x86_fp80 0xK400CB1DC000000000000, metadata !"ogt", metadata !"fpexcept.strict")
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_3 = call x86_fp80 @exp2l(x86_fp80 %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_4 = call x86_fp80 @sinhl(x86_fp80 %value) strictfp
; CHECK: [[COND1:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f80(x86_fp80 [[VALUE]], x86_fp80 0xKC00CB174000000000000, metadata !"olt", metadata !"fpexcept.strict")
; CHECK: [[COND2:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f80(x86_fp80 [[VALUE]], x86_fp80 0xK400CB174000000000000, metadata !"ogt", metadata !"fpexcept.strict")
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_4 = call x86_fp80 @sinhl(x86_fp80 %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_5 = call x86_fp80 @expm1l(x86_fp80 %value) strictfp
; CHECK: [[COND:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f80(x86_fp80 [[VALUE]], x86_fp80 0xK400CB170000000000000, metadata !"ogt", metadata !"fpexcept.strict")
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_5 = call x86_fp80 @expm1l(x86_fp80 %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  ret void
}

declare x86_fp80 @coshl(x86_fp80)
declare x86_fp80 @expl(x86_fp80)
declare x86_fp80 @exp10l(x86_fp80)
declare x86_fp80 @exp2l(x86_fp80)
declare x86_fp80 @sinhl(x86_fp80)
declare x86_fp80 @expm1l(x86_fp80)

define void @test_domain_error(x86_fp80 %value) {
entry:
  %call_00 = call x86_fp80 @acosl(x86_fp80 %value)
; CHECK: [[COND1:%[0-9]+]] = fcmp ogt x86_fp80 %value, 0xK3FFF8000000000000000
; CHECK: [[COND2:%[0-9]+]] = fcmp olt x86_fp80 %value, 0xKBFFF8000000000000000
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_00 = call x86_fp80 @acosl(x86_fp80 %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_01 = call x86_fp80 @asinl(x86_fp80 %value)
; CHECK: [[COND1:%[0-9]+]] = fcmp ogt x86_fp80 %value, 0xK3FFF8000000000000000
; CHECK: [[COND2:%[0-9]+]] = fcmp olt x86_fp80 %value, 0xKBFFF8000000000000000
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_01 = call x86_fp80 @asinl(x86_fp80 %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_02 = call x86_fp80 @cosl(x86_fp80 %value)
; CHECK: [[COND1:%[0-9]+]] = fcmp oeq x86_fp80 %value, 0xKFFFF8000000000000000
; CHECK: [[COND2:%[0-9]+]] = fcmp oeq x86_fp80 %value, 0xK7FFF8000000000000000
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_02 = call x86_fp80 @cosl(x86_fp80 %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_03 = call x86_fp80 @sinl(x86_fp80 %value)
; CHECK: [[COND1:%[0-9]+]] = fcmp oeq x86_fp80 %value, 0xKFFFF8000000000000000
; CHECK: [[COND2:%[0-9]+]] = fcmp oeq x86_fp80 %value, 0xK7FFF8000000000000000
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_03 = call x86_fp80 @sinl(x86_fp80 %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_04 = call x86_fp80 @acoshl(x86_fp80 %value)
; CHECK: [[COND:%[0-9]+]] = fcmp olt x86_fp80 %value, 0xK3FFF8000000000000000
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_04 = call x86_fp80 @acoshl(x86_fp80 %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_05 = call x86_fp80 @sqrtl(x86_fp80 %value)
; CHECK: [[COND:%[0-9]+]] = fcmp olt x86_fp80 %value, 0xK00000000000000000000
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_05 = call x86_fp80 @sqrtl(x86_fp80 %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_06 = call x86_fp80 @atanhl(x86_fp80 %value)
; CHECK: [[COND1:%[0-9]+]] = fcmp oge x86_fp80 %value, 0xK3FFF8000000000000000
; CHECK: [[COND2:%[0-9]+]] = fcmp ole x86_fp80 %value, 0xKBFFF8000000000000000
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_06 = call x86_fp80 @atanhl(x86_fp80 %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_07 = call x86_fp80 @logl(x86_fp80 %value)
; CHECK: [[COND:%[0-9]+]] = fcmp ole x86_fp80 %value, 0xK00000000000000000000
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_07 = call x86_fp80 @logl(x86_fp80 %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_08 = call x86_fp80 @log10l(x86_fp80 %value)
; CHECK: [[COND:%[0-9]+]] = fcmp ole x86_fp80 %value, 0xK00000000000000000000
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_08 = call x86_fp80 @log10l(x86_fp80 %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_09 = call x86_fp80 @log2l(x86_fp80 %value)
; CHECK: [[COND:%[0-9]+]] = fcmp ole x86_fp80 %value, 0xK00000000000000000000
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_09 = call x86_fp80 @log2l(x86_fp80 %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_10 = call x86_fp80 @logbl(x86_fp80 %value)
; CHECK: [[COND:%[0-9]+]] = fcmp ole x86_fp80 %value, 0xK00000000000000000000
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_10 = call x86_fp80 @logbl(x86_fp80 %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_11 = call x86_fp80 @log1pl(x86_fp80 %value)
; CHECK: [[COND:%[0-9]+]] = fcmp ole x86_fp80 %value, 0xKBFFF8000000000000000
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_11 = call x86_fp80 @log1pl(x86_fp80 %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  ret void
}

define void @test_domain_error_strictfp(x86_fp80 %value) strictfp {
entry:
  %call_00 = call x86_fp80 @acosl(x86_fp80 %value) strictfp
; CHECK: [[COND1:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f80(x86_fp80 [[VALUE:%.*]], x86_fp80 0xK3FFF8000000000000000, metadata !"ogt", metadata !"fpexcept.strict")
; CHECK: [[COND2:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f80(x86_fp80 [[VALUE]], x86_fp80 0xKBFFF8000000000000000, metadata !"olt", metadata !"fpexcept.strict")
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_00 = call x86_fp80 @acosl(x86_fp80 %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_01 = call x86_fp80 @asinl(x86_fp80 %value) strictfp
; CHECK: [[COND1:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f80(x86_fp80 [[VALUE]], x86_fp80 0xK3FFF8000000000000000, metadata !"ogt", metadata !"fpexcept.strict")
; CHECK: [[COND2:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f80(x86_fp80 [[VALUE]], x86_fp80 0xKBFFF8000000000000000, metadata !"olt", metadata !"fpexcept.strict")
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_01 = call x86_fp80 @asinl(x86_fp80 %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_02 = call x86_fp80 @cosl(x86_fp80 %value) strictfp
; CHECK: [[COND1:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f80(x86_fp80 [[VALUE]], x86_fp80 0xKFFFF8000000000000000, metadata !"oeq", metadata !"fpexcept.strict")
; CHECK: [[COND2:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f80(x86_fp80 [[VALUE]], x86_fp80 0xK7FFF8000000000000000, metadata !"oeq", metadata !"fpexcept.strict")
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_02 = call x86_fp80 @cosl(x86_fp80 %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_03 = call x86_fp80 @sinl(x86_fp80 %value) strictfp
; CHECK: [[COND1:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f80(x86_fp80 [[VALUE]], x86_fp80 0xKFFFF8000000000000000, metadata !"oeq", metadata !"fpexcept.strict")
; CHECK: [[COND2:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f80(x86_fp80 [[VALUE]], x86_fp80 0xK7FFF8000000000000000, metadata !"oeq", metadata !"fpexcept.strict")
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_03 = call x86_fp80 @sinl(x86_fp80 %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_04 = call x86_fp80 @acoshl(x86_fp80 %value) strictfp
; CHECK: [[COND:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f80(x86_fp80 [[VALUE]], x86_fp80 0xK3FFF8000000000000000, metadata !"olt", metadata !"fpexcept.strict")
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_04 = call x86_fp80 @acoshl(x86_fp80 %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_05 = call x86_fp80 @sqrtl(x86_fp80 %value) strictfp
; CHECK: [[COND:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f80(x86_fp80 [[VALUE]], x86_fp80 0xK00000000000000000000, metadata !"olt", metadata !"fpexcept.strict")
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_05 = call x86_fp80 @sqrtl(x86_fp80 %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_06 = call x86_fp80 @atanhl(x86_fp80 %value) strictfp
; CHECK: [[COND1:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f80(x86_fp80 [[VALUE]], x86_fp80 0xK3FFF8000000000000000, metadata !"oge", metadata !"fpexcept.strict")
; CHECK: [[COND2:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f80(x86_fp80 [[VALUE]], x86_fp80 0xKBFFF8000000000000000, metadata !"ole", metadata !"fpexcept.strict")
; CHECK: [[COND:%[0-9]+]] = or i1 [[COND2]], [[COND1]]
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_06 = call x86_fp80 @atanhl(x86_fp80 %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_07 = call x86_fp80 @logl(x86_fp80 %value) strictfp
; CHECK: [[COND:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f80(x86_fp80 [[VALUE]], x86_fp80 0xK00000000000000000000, metadata !"ole", metadata !"fpexcept.strict")
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_07 = call x86_fp80 @logl(x86_fp80 %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_08 = call x86_fp80 @log10l(x86_fp80 %value) strictfp
; CHECK: [[COND:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f80(x86_fp80 [[VALUE]], x86_fp80 0xK00000000000000000000, metadata !"ole", metadata !"fpexcept.strict")
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_08 = call x86_fp80 @log10l(x86_fp80 %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_09 = call x86_fp80 @log2l(x86_fp80 %value) strictfp
; CHECK: [[COND:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f80(x86_fp80 [[VALUE]], x86_fp80 0xK00000000000000000000, metadata !"ole", metadata !"fpexcept.strict")
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_09 = call x86_fp80 @log2l(x86_fp80 %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_10 = call x86_fp80 @logbl(x86_fp80 %value) strictfp
; CHECK: [[COND:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f80(x86_fp80 [[VALUE]], x86_fp80 0xK00000000000000000000, metadata !"ole", metadata !"fpexcept.strict")
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_10 = call x86_fp80 @logbl(x86_fp80 %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  %call_11 = call x86_fp80 @log1pl(x86_fp80 %value) strictfp
; CHECK: [[COND:%[0-9]+]] = call i1 @llvm.experimental.constrained.fcmp.f80(x86_fp80 [[VALUE]], x86_fp80 0xKBFFF8000000000000000, metadata !"ole", metadata !"fpexcept.strict")
; CHECK: br i1 [[COND]], label %[[CALL_LABEL:cdce.call[0-9]*]], label %[[END_LABEL:cdce.end[0-9]*]], !prof ![[BRANCH_WEIGHT]]
; CHECK: [[CALL_LABEL]]:
; CHECK-NEXT: %call_11 = call x86_fp80 @log1pl(x86_fp80 %value)
; CHECK-NEXT: br label %[[END_LABEL]]
; CHECK: [[END_LABEL]]:

  ret void
}

declare x86_fp80 @acosl(x86_fp80)
declare x86_fp80 @asinl(x86_fp80)
declare x86_fp80 @cosl(x86_fp80)
declare x86_fp80 @sinl(x86_fp80)
declare x86_fp80 @acoshl(x86_fp80)
declare x86_fp80 @sqrtl(x86_fp80)
declare x86_fp80 @atanhl(x86_fp80)
declare x86_fp80 @logl(x86_fp80)
declare x86_fp80 @log10l(x86_fp80)
declare x86_fp80 @log2l(x86_fp80)
declare x86_fp80 @logbl(x86_fp80)
declare x86_fp80 @log1pl(x86_fp80)

; CHECK: ![[BRANCH_WEIGHT]] = !{!"branch_weights", i32 1, i32 2000}

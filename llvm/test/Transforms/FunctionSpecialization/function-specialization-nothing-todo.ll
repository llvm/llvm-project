; REQUIRES: asserts
; RUN: opt -function-specialization -debug -S < %s 2>&1 | FileCheck %s

; The purpose of this test is to check that we don't run the solver as there's
; nothing to do here. For a test that doesn't trigger function specialisation,
; it is intentionally 'big' because we also want to check that the ssa.copy
; intrinsics that are introduced by the solver are cleaned up if we bail
; early. Thus, first check the debug messages for the introduction of these
; intrinsics:

; CHECK: FnSpecialization: Analysing decl: foo
; CHECK: Found replacement{{.*}} call i32 @llvm.ssa.copy.i32
; CHECK: Found replacement{{.*}} call i32 @llvm.ssa.copy.i32

; Then, make sure the solver didn't run:

; CHECK-NOT: Running solver

; Finally, check the absence and thus removal of these intrinsics:

; CHECK-LABEL: @foo
; CHECK-NOT:   call i32 @llvm.ssa.copy.i32

@N = external dso_local global i32, align 4
@B = external dso_local global ptr, align 8
@A = external dso_local global ptr, align 8

define dso_local i32 @foo() {
entry:
  br label %for.cond

for.cond:
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %0 = load i32, ptr @N, align 4
  %cmp = icmp slt i32 %i.0, %0
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret i32 undef

for.body:
  %1 = load ptr, ptr @B, align 8
  %idxprom = sext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds i32, ptr %1, i64 %idxprom
  %2 = load i32, ptr %arrayidx, align 4
  %3 = load ptr, ptr @A, align 8
  %arrayidx2 = getelementptr inbounds i32, ptr %3, i64 %idxprom
  store i32 %2, ptr %arrayidx2, align 4
  %inc = add nsw i32 %i.0, 1
  br label %for.cond
}

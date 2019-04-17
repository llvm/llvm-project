; RUN: opt < %s -jump-threading -dce -S | FileCheck %s

declare void @llvm.experimental.guard(i1, ...)

declare i32 @f1()
declare i32 @f2()

define i32 @branch_implies_guard(i32 %a) {
; CHECK-LABEL: @branch_implies_guard(
  %cond = icmp slt i32 %a, 10
  br i1 %cond, label %T1, label %F1

T1:
; CHECK:       T1.split
; CHECK:         %v1 = call i32 @f1()
; CHECK-NEXT:    %retVal
; CHECK-NEXT:    br label %Merge
  %v1 = call i32 @f1()
  br label %Merge

F1:
; CHECK:       F1.split
; CHECK:         %v2 = call i32 @f2()
; CHECK-NEXT:    %retVal
; CHECK-NEXT:    %condGuard
; CHECK-NEXT:    call void (i1, ...) @llvm.experimental.guard(i1 %condGuard
; CHECK-NEXT:    br label %Merge
  %v2 = call i32 @f2()
  br label %Merge

Merge:
; CHECK:       Merge
; CHECK-NOT:     call void(i1, ...) @llvm.experimental.guard(
  %retPhi = phi i32 [ %v1, %T1 ], [ %v2, %F1 ]
  %retVal = add i32 %retPhi, 10
  %condGuard = icmp slt i32 %a, 20
  call void(i1, ...) @llvm.experimental.guard(i1 %condGuard) [ "deopt"() ]
  ret i32 %retVal
}

define i32 @not_branch_implies_guard(i32 %a) {
; CHECK-LABEL: @not_branch_implies_guard(
  %cond = icmp slt i32 %a, 20
  br i1 %cond, label %T1, label %F1

T1:
; CHECK:       T1.split:
; CHECK-NEXT:    %v1 = call i32 @f1()
; CHECK-NEXT:    %retVal
; CHECK-NEXT:    %condGuard
; CHECK-NEXT:    call void (i1, ...) @llvm.experimental.guard(i1 %condGuard
; CHECK-NEXT:    br label %Merge
  %v1 = call i32 @f1()
  br label %Merge

F1:
; CHECK:       F1.split:
; CHECK-NEXT:   %v2 = call i32 @f2()
; CHECK-NEXT:   %retVal
; CHECK-NEXT:   br label %Merge
  %v2 = call i32 @f2()
  br label %Merge

Merge:
; CHECK:       Merge
; CHECK-NOT:     call void(i1, ...) @llvm.experimental.guard(
  %retPhi = phi i32 [ %v1, %T1 ], [ %v2, %F1 ]
  %retVal = add i32 %retPhi, 10
  %condGuard = icmp sgt i32 %a, 10
  call void(i1, ...) @llvm.experimental.guard(i1 %condGuard) [ "deopt"() ]
  ret i32 %retVal
}

define i32 @branch_overlaps_guard(i32 %a) {
; CHECK-LABEL: @branch_overlaps_guard(
  %cond = icmp slt i32 %a, 20
  br i1 %cond, label %T1, label %F1

T1:
; CHECK:        T1:
; CHECK-NEXT:      %v1 = call i32 @f1()
; CHECK-NEXT:      br label %Merge
  %v1 = call i32 @f1()
  br label %Merge

F1:
; CHECK:        F1:
; CHECK-NEXT:     %v2 = call i32 @f2()
; CHECK-NEXT:     br label %Merge
  %v2 = call i32 @f2()
  br label %Merge

Merge:
; CHECK:       Merge
; CHECK:         %condGuard = icmp slt i32 %a, 10
; CHECK-NEXT:    call void (i1, ...) @llvm.experimental.guard(i1 %condGuard) [ "deopt"() ]
  %retPhi = phi i32 [ %v1, %T1 ], [ %v2, %F1 ]
  %retVal = add i32 %retPhi, 10
  %condGuard = icmp slt i32 %a, 10
  call void(i1, ...) @llvm.experimental.guard(i1 %condGuard) [ "deopt"() ]
  ret i32 %retVal
}

define i32 @branch_doesnt_overlap_guard(i32 %a) {
; CHECK-LABEL: @branch_doesnt_overlap_guard(
  %cond = icmp slt i32 %a, 10
  br i1 %cond, label %T1, label %F1

T1:
; CHECK:        T1:
; CHECK-NEXT:      %v1 = call i32 @f1()
; CHECK-NEXT:      br label %Merge
  %v1 = call i32 @f1()
  br label %Merge

F1:
; CHECK:        F1:
; CHECK-NEXT:     %v2 = call i32 @f2()
; CHECK-NEXT:     br label %Merge
  %v2 = call i32 @f2()
  br label %Merge

Merge:
; CHECK:       Merge
; CHECK:         %condGuard = icmp sgt i32 %a, 20
; CHECK-NEXT:    call void (i1, ...) @llvm.experimental.guard(i1 %condGuard) [ "deopt"() ]
  %retPhi = phi i32 [ %v1, %T1 ], [ %v2, %F1 ]
  %retVal = add i32 %retPhi, 10
  %condGuard = icmp sgt i32 %a, 20
  call void(i1, ...) @llvm.experimental.guard(i1 %condGuard) [ "deopt"() ]
  ret i32 %retVal
}

define i32 @not_a_diamond1(i32 %a, i1 %cond1) {
; CHECK-LABEL: @not_a_diamond1(
  br i1 %cond1, label %Pred, label %Exit

Pred:
; CHECK:       Pred:
; CHECK-NEXT:    switch i32 %a, label %Exit
  switch i32 %a, label %Exit [
    i32 10, label %Merge
    i32 20, label %Merge
  ]

Merge:
; CHECK:       Merge:
; CHECK-NEXT:    call void (i1, ...) @llvm.experimental.guard(i1 %cond1) [ "deopt"() ]
; CHECK-NEXT:    br label %Exit
  call void(i1, ...) @llvm.experimental.guard(i1 %cond1) [ "deopt"() ]
  br label %Exit

Exit:
; CHECK:       Exit:
; CHECK-NEXT:    ret i32 %a
  ret i32 %a
}

define void @not_a_diamond2(i32 %a, i1 %cond1) {
; CHECK-LABEL: @not_a_diamond2(
  br label %Parent

Merge:
  call void(i1, ...) @llvm.experimental.guard(i1 %cond1)[ "deopt"() ]
  ret void

Pred:
; CHECK-NEXT:  Pred:
; CHECK-NEXT:    switch i32 %a, label %Exit
  switch i32 %a, label %Exit [
    i32 10, label %Merge
    i32 20, label %Merge
  ]

Parent:
  br label %Pred

Exit:
; CHECK:       Merge:
; CHECK-NEXT:    call void (i1, ...) @llvm.experimental.guard(i1 %cond1) [ "deopt"() ]
; CHECK-NEXT:    ret void
  ret void
}

declare void @never_called(i1)

; LVI uses guard to identify value of %c2 in branch as true, we cannot replace that
; guard with guard(true & c1).
define void @dont_fold_guard(i8* %addr, i32 %i, i32 %length) {
; CHECK-LABEL: dont_fold_guard
; CHECK: %wide.chk = and i1 %c1, %c2
; CHECK-NEXT: experimental.guard(i1 %wide.chk)
; CHECK-NEXT: call void @never_called(i1 true)
; CHECK-NEXT: ret void
  %c1 = icmp ult i32 %i, %length
  %c2 = icmp eq i32 %i, 0
  %wide.chk = and i1 %c1, %c2
  call void(i1, ...) @llvm.experimental.guard(i1 %wide.chk) [ "deopt"() ]
  br i1 %c2, label %BB1, label %BB2

BB1:
  call void @never_called(i1 %c2)
  ret void

BB2:
  ret void
}

declare void @dummy(i1) nounwind argmemonly
; same as dont_fold_guard1 but there's a use immediately after guard and before
; branch. We can fold that use.
define void @dont_fold_guard2(i8* %addr, i32 %i, i32 %length) {
; CHECK-LABEL: dont_fold_guard2
; CHECK: %wide.chk = and i1 %c1, %c2
; CHECK-NEXT: experimental.guard(i1 %wide.chk)
; CHECK-NEXT: dummy(i1 true)
; CHECK-NEXT: call void @never_called(i1 true)
; CHECK-NEXT: ret void
  %c1 = icmp ult i32 %i, %length
  %c2 = icmp eq i32 %i, 0
  %wide.chk = and i1 %c1, %c2
  call void(i1, ...) @llvm.experimental.guard(i1 %wide.chk) [ "deopt"() ]
  call void @dummy(i1 %c2)
  br i1 %c2, label %BB1, label %BB2

BB1:
  call void @never_called(i1 %c2)
  ret void

BB2:
  ret void
}

; same as dont_fold_guard1 but condition %cmp is not an instruction.
; We cannot fold the guard under any circumstance.
; FIXME: We can merge unreachableBB2 into not_zero.
define void @dont_fold_guard3(i8* %addr, i1 %cmp, i32 %i, i32 %length) {
; CHECK-LABEL: dont_fold_guard3
; CHECK: guard(i1 %cmp)
  call void(i1, ...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
  br i1 %cmp, label %BB1, label %BB2

BB1:
  call void @never_called(i1 %cmp)
  ret void

BB2:
  ret void
}

declare void @f(i1)
; Same as dont_fold_guard1 but use switch instead of branch.
; triggers source code `ProcessThreadableEdges`.
define void @dont_fold_guard4(i1 %cmp1, i32 %i) nounwind {
; CHECK-LABEL: dont_fold_guard4 
; CHECK-LABEL: L2:
; CHECK-NEXT: %cmp = icmp eq i32 %i, 0 
; CHECK-NEXT: guard(i1 %cmp)
; CHECK-NEXT: dummy(i1 true)
; CHECK-NEXT: @f(i1 true)
; CHECK-NEXT: ret void
entry:
  br i1 %cmp1, label %L0, label %L3 
L0:
  %cmp = icmp eq i32 %i, 0
  call void(i1, ...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
  call void @dummy(i1 %cmp)
  switch i1 %cmp, label %L3 [
    i1 false, label %L1
    i1 true, label %L2
    ]

L1:
  ret void
L2:
  call void @f(i1 %cmp)
  ret void
L3:
  ret void
}

; Make sure that we don't PRE a non-speculable load across a guard.
define void @unsafe_pre_across_guard(i8* %p, i1 %load.is.valid) {

; CHECK-LABEL: @unsafe_pre_across_guard(
; CHECK-NOT:   loaded.pr
; CHECK:       entry:
; CHECK-NEXT:    br label %loop
; CHECK:       loop:
; CHECK-NEXT:    call void (i1, ...) @llvm.experimental.guard(i1 %load.is.valid) [ "deopt"() ]
; CHECK-NEXT:    %loaded = load i8, i8* %p
; CHECK-NEXT:    %continue = icmp eq i8 %loaded, 0
; CHECK-NEXT:    br i1 %continue, label %exit, label %loop
entry:
  br label %loop

loop:                                             ; preds = %loop, %entry
  call void (i1, ...) @llvm.experimental.guard(i1 %load.is.valid) [ "deopt"() ]
  %loaded = load i8, i8* %p
  %continue = icmp eq i8 %loaded, 0
  br i1 %continue, label %exit, label %loop

exit:                                             ; preds = %loop
  ret void
}

; Make sure that we can safely PRE a speculable load across a guard.
define void @safe_pre_across_guard(i8* noalias nocapture readonly dereferenceable(8) %p, i1 %load.is.valid) {

; CHECK-LABEL: @safe_pre_across_guard(
; CHECK:       entry:
; CHECK-NEXT:    %loaded.pr = load i8, i8* %p
; CHECK-NEXT:    br label %loop
; CHECK:       loop:
; CHECK-NEXT:    %loaded = phi i8 [ %loaded, %loop ], [ %loaded.pr, %entry ]
; CHECK-NEXT:    call void (i1, ...) @llvm.experimental.guard(i1 %load.is.valid) [ "deopt"() ]
; CHECK-NEXT:    %continue = icmp eq i8 %loaded, 0
; CHECK-NEXT:    br i1 %continue, label %exit, label %loop

entry:
  br label %loop

loop:                                             ; preds = %loop, %entry
  call void (i1, ...) @llvm.experimental.guard(i1 %load.is.valid) [ "deopt"() ]
  %loaded = load i8, i8* %p
  %continue = icmp eq i8 %loaded, 0
  br i1 %continue, label %exit, label %loop

exit:                                             ; preds = %loop
  ret void
}

; Make sure that we don't PRE a non-speculable load across a call which may
; alias with the load.
define void @unsafe_pre_across_call(i8* %p) {

; CHECK-LABEL: @unsafe_pre_across_call(
; CHECK-NOT:   loaded.pr
; CHECK:       entry:
; CHECK-NEXT:    br label %loop
; CHECK:       loop:
; CHECK-NEXT:    call i32 @f1()
; CHECK-NEXT:    %loaded = load i8, i8* %p
; CHECK-NEXT:    %continue = icmp eq i8 %loaded, 0
; CHECK-NEXT:    br i1 %continue, label %exit, label %loop
entry:
  br label %loop

loop:                                             ; preds = %loop, %entry
  call i32 @f1()
  %loaded = load i8, i8* %p
  %continue = icmp eq i8 %loaded, 0
  br i1 %continue, label %exit, label %loop

exit:                                             ; preds = %loop
  ret void
}

; Make sure that we can safely PRE a speculable load across a call.
define void @safe_pre_across_call(i8* noalias nocapture readonly dereferenceable(8) %p) {

; CHECK-LABEL: @safe_pre_across_call(
; CHECK:       entry:
; CHECK-NEXT:    %loaded.pr = load i8, i8* %p
; CHECK-NEXT:    br label %loop
; CHECK:       loop:
; CHECK-NEXT:    %loaded = phi i8 [ %loaded, %loop ], [ %loaded.pr, %entry ]
; CHECK-NEXT:    call i32 @f1()
; CHECK-NEXT:    %continue = icmp eq i8 %loaded, 0
; CHECK-NEXT:    br i1 %continue, label %exit, label %loop

entry:
  br label %loop

loop:                                             ; preds = %loop, %entry
  call i32 @f1()
  %loaded = load i8, i8* %p
  %continue = icmp eq i8 %loaded, 0
  br i1 %continue, label %exit, label %loop

exit:                                             ; preds = %loop
  ret void
}

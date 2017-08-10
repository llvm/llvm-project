; PR23538
; RUN: opt < %s -indvars -loop-deletion -S | FileCheck %s

; Check IndVarSimplify should not replace exit value because or else
; udiv will be introduced by expand and the cost will be high.

declare void @_Z3mixRjj(i32* dereferenceable(4), i32)
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)

define i32 @_Z3fooPKcjj(i8* nocapture readonly %s, i32 %len, i32 %c) {
; CHECK-LABEL: @_Z3fooPKcjj(
; CHECK-NOT: udiv
entry:
  %a = alloca i32, align 4
  %tmp = bitcast i32* %a to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %tmp)
  store i32 -1640531527, i32* %a, align 4
  %cmp8 = icmp ugt i32 %len, 11
  br i1 %cmp8, label %while.body.lr.ph, label %while.end

while.body.lr.ph:                                 ; preds = %entry
  br label %while.body

while.body:                                       ; preds = %while.body, %while.body.lr.ph
  %keylen.010 = phi i32 [ %len, %while.body.lr.ph ], [ %sub, %while.body ]
  %s.addr.09 = phi i8* [ %s, %while.body.lr.ph ], [ %add.ptr, %while.body ]
  %tmp1 = bitcast i8* %s.addr.09 to i32*
  %tmp2 = load i32, i32* %tmp1, align 4
  %shl.i = shl i32 %tmp2, 1
  %and.i = and i32 %shl.i, 16843008
  %tmp3 = load i32, i32* %a, align 4
  %sub.i = add i32 %tmp3, %tmp2
  %add = sub i32 %sub.i, %and.i
  store i32 %add, i32* %a, align 4
  %add.ptr = getelementptr inbounds i8, i8* %s.addr.09, i64 12
  %sub = add i32 %keylen.010, -12
  %cmp = icmp ugt i32 %sub, 11
  br i1 %cmp, label %while.body, label %while.cond.while.end_crit_edge

while.cond.while.end_crit_edge:                   ; preds = %while.body
  %sub.lcssa = phi i32 [ %sub, %while.body ]
  br label %while.end

while.end:                                        ; preds = %while.cond.while.end_crit_edge, %entry
  %keylen.0.lcssa = phi i32 [ %sub.lcssa, %while.cond.while.end_crit_edge ], [ %len, %entry ]
  call void @_Z3mixRjj(i32* dereferenceable(4) %a, i32 %keylen.0.lcssa)
  %tmp4 = load i32, i32* %a, align 4
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %tmp)
  ret i32 %tmp4
}

define i32 @zero_backedge_count_test(i32 %unknown_init, i32* %unknown_mem) {
; CHECK-LABEL: @zero_backedge_count_test(
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry], [ %iv.inc, %loop ]
  %unknown_phi = phi i32 [ %unknown_init, %entry ], [ %unknown_next, %loop ]
  %iv.inc = add i32 %iv, 1
  %be_taken = icmp ne i32 %iv.inc, 1
  %unknown_next = load volatile i32, i32* %unknown_mem
  br i1 %be_taken, label %loop, label %leave

leave:
; We can fold %unknown_phi even though the backedge value for it is completely
; unknown, since we can prove that the loop's backedge taken count is 0.

; CHECK: leave:
; CHECK: ret i32 %unknown_init
  %exit_val = phi i32 [ %unknown_phi, %loop ]
  ret i32 %exit_val
}

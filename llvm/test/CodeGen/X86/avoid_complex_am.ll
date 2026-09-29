; RUN: opt -S -loop-reduce < %s | FileCheck %s
; Complex addressing mode are costly.
; Make loop-reduce prefer unscaled accesses.
; On X86, reg1 + 1*reg2 has the same cost as reg1 + 8*reg2.
; Therefore, LSR currently prefers to fold as much computation as possible
; in the addressing mode.
; <rdar://problem/16730541>
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx"

define void @mulDouble(ptr nocapture %a, ptr nocapture %b, ptr nocapture %c) {
; CHECK: @mulDouble
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
; CHECK: [[IV:%[^ ]+]] = phi i64 [ [[IVNEXT:%[^,]+]], %for.body ], [ 0, %entry ]
; Only one induction variable should have been generated.
; CHECK-NOT: phi
  %indvars.iv = phi i64 [ 1, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = add nsw i64 %indvars.iv, -1
  %arrayidx = getelementptr inbounds double, ptr %b, i64 %tmp
  %tmp1 = load double, ptr %arrayidx, align 8
; The induction variable should carry the scaling factor: 1.
; CHECK: [[IVNEXT]] = add nuw nsw i64 [[IV]], 1
  %indvars.iv.next = add i64 %indvars.iv, 1
  %arrayidx2 = getelementptr inbounds double, ptr %c, i64 %indvars.iv.next
  %tmp2 = load double, ptr %arrayidx2, align 8
  %mul = fmul double %tmp1, %tmp2
  %arrayidx4 = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  store double %mul, ptr %arrayidx4, align 8
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
; Comparison should be 19 * 1 = 19.
; CHECK: icmp eq i32 {{%[^,]+}}, 19
  %exitcond = icmp eq i32 %lftr.wideiv, 20
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

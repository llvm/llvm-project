; RUN: llc < %s -mtriple=x86_64-- -mcpu=corei7 | FileCheck %s
; PR1198

define i64 @foo(i64 %x, i64 %y) {
        %tmp0 = zext i64 %x to i128
        %tmp1 = zext i64 %y to i128
        %tmp2 = mul i128 %tmp0, %tmp1
        %tmp7 = zext i32 64 to i128
        %tmp3 = lshr i128 %tmp2, %tmp7
        %tmp4 = trunc i128 %tmp3 to i64
        ret i64 %tmp4
}

; <rdar://problem/14096009> superfluous multiply by high part of
; zero-extended value.
; CHECK: @mul1
; CHECK-NOT: imulq
; CHECK: mulq
; CHECK-NOT: imulq
define i64 @mul1(i64 %n, i64* nocapture %z, i64* nocapture %x, i64 %y) {
entry:
  %conv = zext i64 %y to i128
  %cmp11 = icmp eq i64 %n, 0
  br i1 %cmp11, label %for.end, label %for.body

for.body:                                         ; preds = %entry, %for.body
  %carry.013 = phi i64 [ %conv6, %for.body ], [ 0, %entry ]
  %i.012 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i64, i64* %x, i64 %i.012
  %0 = load i64, i64* %arrayidx, align 8
  %conv2 = zext i64 %0 to i128
  %mul = mul i128 %conv2, %conv
  %conv3 = zext i64 %carry.013 to i128
  %add = add i128 %mul, %conv3
  %conv4 = trunc i128 %add to i64
  %arrayidx5 = getelementptr inbounds i64, i64* %z, i64 %i.012
  store i64 %conv4, i64* %arrayidx5, align 8
  %shr = lshr i128 %add, 64
  %conv6 = trunc i128 %shr to i64
  %inc = add i64 %i.012, 1
  %exitcond = icmp eq i64 %inc, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret i64 0
}

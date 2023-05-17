; RUN: opt < %s -passes=indvars -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n32:64"

; Indvars should be able to fold IV increments into shr when low bits are zero.
;
; CHECK-LABEL: @foldIncShr(
; CHECK: shr.1 = lshr i32 %0, 5
define i32 @foldIncShr(ptr %bitmap, i32 %bit_addr, i32 %nbits) nounwind {
entry:
  br label %while.body

while.body:
  %0 = phi i32 [ 0, %entry ], [ %inc.2, %while.body ]
  %shr = lshr i32 %0, 5
  %arrayidx = getelementptr inbounds i32, ptr %bitmap, i32 %shr
  %tmp6 = load i32, ptr %arrayidx, align 4
  %inc.1 = add i32 %0, 1
  %shr.1 = lshr i32 %inc.1, 5
  %arrayidx.1 = getelementptr inbounds i32, ptr %bitmap, i32 %shr.1
  %tmp6.1 = load i32, ptr %arrayidx.1, align 4
  %inc.2 = add i32 %inc.1, 1
  %exitcond.3 = icmp eq i32 %inc.2, 128
  br i1 %exitcond.3, label %while.end, label %while.body

while.end:
  %r = add i32 %tmp6, %tmp6.1
  ret i32 %r
}

; Invdars should not fold an increment into shr unless 2^shiftBits is
; a multiple of the recurrence step.
;
; CHECK-LABEL: @noFoldIncShr(
; CHECK: shr.1 = lshr i32 %inc.1, 5
define i32 @noFoldIncShr(ptr %bitmap, i32 %bit_addr, i32 %nbits) nounwind {
entry:
  br label %while.body

while.body:
  %0 = phi i32 [ 0, %entry ], [ %inc.3, %while.body ]
  %shr = lshr i32 %0, 5
  %arrayidx = getelementptr inbounds i32, ptr %bitmap, i32 %shr
  %tmp6 = load i32, ptr %arrayidx, align 4
  %inc.1 = add i32 %0, 1
  %shr.1 = lshr i32 %inc.1, 5
  %arrayidx.1 = getelementptr inbounds i32, ptr %bitmap, i32 %shr.1
  %tmp6.1 = load i32, ptr %arrayidx.1, align 4
  %inc.3 = add i32 %inc.1, 2
  %exitcond.3 = icmp eq i32 %inc.3, 96
  br i1 %exitcond.3, label %while.end, label %while.body

while.end:
  %r = add i32 %tmp6, %tmp6.1
  ret i32 %r
}

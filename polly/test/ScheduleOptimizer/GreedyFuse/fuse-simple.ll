; RUN: opt %loadNPMPolly -polly-reschedule=0 -polly-loopfusion-greedy=1 -polly-postopts=0 '-passes=print<polly-opt-isl>' -disable-output < %s | FileCheck %s
; RUN: opt %loadNPMPolly -polly-reschedule=1 -polly-loopfusion-greedy=1 -polly-postopts=0 '-passes=print<polly-opt-isl>' -disable-output < %s | FileCheck %s

define void @func(i32 %n, ptr noalias nonnull %A) {
entry:
  br label %for1

for1:
  %j1 = phi i32 [0, %entry], [%j1.inc, %inc1]
  %j1.cmp = icmp slt i32 %j1, %n
  br i1 %j1.cmp, label %body1, label %exit1

    body1:
      %arrayidx1 = getelementptr inbounds double, ptr %A, i32 %j1
      store double 21.0, ptr %arrayidx1
      br label %inc1

inc1:
  %j1.inc = add nuw nsw i32 %j1, 1
  br label %for1

exit1:
  br label %for2

for2:
  %j2 = phi i32 [0, %exit1], [%j2.inc, %inc2]
  %j2.cmp = icmp slt i32 %j2, %n
  br i1 %j2.cmp, label %body2, label %exit2

    body2:
      %arrayidx2 = getelementptr inbounds double, ptr %A, i32 %j2
      store double 42.0, ptr %arrayidx2
      br label %inc2

inc2:
  %j2.inc = add nuw nsw i32 %j2, 1
  br label %for2

exit2:
  br label %return

return:
  ret void
}


; CHECK:      Calculated schedule:
; CHECK-NEXT: domain: "[n] -> { Stmt_body2[i0] : 0 <= i0 < n; Stmt_body1[i0] : 0 <= i0 < n }"
; CHECK-NEXT: child:
; CHECK-NEXT:   schedule: "[n] -> [{ Stmt_body2[i0] -> [(i0)]; Stmt_body1[i0] -> [(i0)] }]"
; CHECK-NEXT:   child:
; CHECK-NEXT:     sequence:
; CHECK-NEXT:     - filter: "[n] -> { Stmt_body1[i0] }"
; CHECK-NEXT:     - filter: "[n] -> { Stmt_body2[i0] }"

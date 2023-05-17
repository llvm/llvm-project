; RUN: opt %loadPolly -polly-reschedule=0 -polly-loopfusion-greedy=1 -polly-postopts=0 -polly-print-opt-isl -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -polly-reschedule=1 -polly-loopfusion-greedy=1 -polly-postopts=0 -polly-print-opt-isl -disable-output < %s | FileCheck %s

define void @func(i32 %n, ptr  noalias nonnull %A,  ptr  noalias nonnull %B) {
entry:
  br label %outer.for1

outer.for1:
  %k1 = phi i32 [0, %entry], [%k1.inc, %outer.inc1]
  %k1.cmp = icmp slt i32 %k1, %n
  br i1 %k1.cmp, label %for1, label %outer.exit1

  for1:
    %j1 = phi i32 [0, %outer.for1], [%j1.inc, %inc1]
    %j1.cmp = icmp slt i32 %j1, %n
    br i1 %j1.cmp, label %body1, label %exit1

      body1:
        %arrayidx1 = getelementptr inbounds [1024 x double], ptr %A, i32 %k1, i32 %j1
        store double 21.0, ptr %arrayidx1
        br label %inc1

  inc1:
    %j1.inc = add nuw nsw i32 %j1, 1
    br label %for1

  exit1:
    br label %outer.inc1

outer.inc1:
  %k1.inc = add nuw nsw i32 %k1, 1
  br label %outer.for1

outer.exit1:
  br label %outer.for2

outer.for2:
  %k2 = phi i32 [0, %outer.exit1], [%k2.inc, %outer.inc2]
  %k2.cmp = icmp slt i32 %k2, %n
  br i1 %k2.cmp, label %for2, label %outer.exit2

  for2:
    %j2 = phi i32 [0, %outer.for2], [%j2.inc, %inc2]
    %j2.cmp = icmp slt i32 %j2, %n
    br i1 %j2.cmp, label %body2, label %exit2

      body2:
        %arrayidx2 = getelementptr inbounds [1024 x double], ptr %A, i32 %k2, i32 %j2
        store double 42.0, ptr %arrayidx2
        br label %inc2

  inc2:
    %j2.inc = add nuw nsw i32 %j2, 1
    br label %for2

  exit2:
    br label %outer.inc2

outer.inc2:
  %k2.inc = add nuw nsw i32 %k2, 1
  br label %outer.for2

outer.exit2:
  br label %return

return:
  ret void
}


; CHECK:      Calculated schedule:
; CHECK-NEXT: domain: "[n] -> { Stmt_body2[i0, i1] : 0 <= i0 < n and 0 <= i1 < n; Stmt_body1[i0, i1] : 0 <= i0 < n and 0 <= i1 < n }"
; CHECK-NEXT: child:
; CHECK-NEXT:   schedule: "[n] -> [{ Stmt_body2[i0, i1] -> [(i0)]; Stmt_body1[i0, i1] -> [(i0)] }, { Stmt_body2[i0, i1] -> [(i1)]; Stmt_body1[i0, i1] -> [(i1)] }]"
; CHECK-NEXT:   child:
; CHECK-NEXT:     sequence:
; CHECK-NEXT:     - filter: "[n] -> { Stmt_body1[i0, i1] }"
; CHECK-NEXT:     - filter: "[n] -> { Stmt_body2[i0, i1] }"

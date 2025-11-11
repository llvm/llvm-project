; RUN: opt %loadNPMPolly -polly-reschedule=0 -polly-loopfusion-greedy=1 -polly-postopts=0 '-passes=print<polly-opt-isl>' -disable-output < %s | FileCheck %s --check-prefixes=CHECK,RAW
; RUN: opt %loadNPMPolly -polly-reschedule=1 -polly-loopfusion-greedy=1 -polly-postopts=0 '-passes=print<polly-opt-isl>' -disable-output < %s | FileCheck %s --check-prefixes=CHECK,OPT

define void @func(i32 %n, ptr noalias nonnull %A) {
entry:
  br label %outer.for

outer.for:
  %k = phi i32 [0, %entry], [%k.inc, %outer.inc]
  %k.cmp = icmp slt i32 %k, %n
  br i1 %k.cmp, label %for1, label %outer.exit

  for1:
    %j1 = phi i32 [0, %outer.for], [%j1.inc, %inc1]
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
    br label %outer.inc

outer.inc:
  %k.inc = add nuw nsw i32 %k, 1
  br label %outer.for

outer.exit:
    br label %return

return:
  ret void
}


; CHECK:      Calculated schedule:
; CHECK-NEXT: domain: "[n] -> { Stmt_body2[i0, i1] : 0 <= i0 < n and 0 <= i1 < n; Stmt_body1[i0, i1] : 0 <= i0 < n and 0 <= i1 < n }"
; CHECK-NEXT: child:
; RAW-NEXT:     schedule: "[n] -> [{ Stmt_body2[i0, i1] -> [(i0)]; Stmt_body1[i0, i1] -> [(i0)] }, { Stmt_body2[i0, i1] -> [(i1)]; Stmt_body1[i0, i1] -> [(i1)] }]"
; OPT-NEXT:     schedule: "[n] -> [{ Stmt_body2[i0, i1] -> [(i1)]; Stmt_body1[i0, i1] -> [(i1)] }, { Stmt_body2[i0, i1] -> [(i0)]; Stmt_body1[i0, i1] -> [(i0)] }]"
; OPT-NEXT:     permutable: 1
; OPT-NEXT:     coincident: [ 1, 0 ]
; CHECK-NEXT:   child:
; CHECK-NEXT:     sequence:
; CHECK-NEXT:     - filter: "[n] -> { Stmt_body1[i0, i1] }"
; CHECK-NEXT:     - filter: "[n] -> { Stmt_body2[i0, i1] }"

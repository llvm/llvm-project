; RUN: opt < %s -S -passes=loop-unroll -unroll-allow-partial -unroll-max-count=1 | FileCheck %s
; Checks that unroll MaxCount is honored.
;
; CHECK-LABEL: @foo(
; CHECK-LABEL: for.body:
; CHECK-NEXT: phi
; CHECK-NEXT: getelementptr
; CHECK-NEXT: load
; CHECK-NEXT: add
; CHECK-NEXT: store
; CHECK-NEXT: add
; CHECK-NEXT: icmp
; CHECK-NEXT: br
define void @foo(i32* nocapture %a) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}


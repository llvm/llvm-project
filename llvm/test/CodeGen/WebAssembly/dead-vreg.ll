; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Check that unused vregs aren't assigned registers.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

define void @foo(i32* nocapture %a, i32 %w, i32 %h) {
; CHECK-LABEL: foo:
; CHECK-NEXT: .param i32, i32, i32
; CHECK-NEXT: .local i32, i32, i32, i32, i32, i32, i32, i32, i32{{$}}
entry:
  %cmp.19 = icmp sgt i32 %h, 0
  br i1 %cmp.19, label %for.cond.1.preheader.lr.ph, label %for.end.7

for.cond.1.preheader.lr.ph:
  %cmp2.17 = icmp sgt i32 %w, 0
  br label %for.cond.1.preheader

for.cond.1.preheader:
  %y.020 = phi i32 [ 0, %for.cond.1.preheader.lr.ph ], [ %inc6, %for.inc.5 ]
  br i1 %cmp2.17, label %for.body.3.lr.ph, label %for.inc.5

for.body.3.lr.ph:
  %mul4 = mul nsw i32 %y.020, %w
  br label %for.body.3

for.body.3:
  %x.018 = phi i32 [ 0, %for.body.3.lr.ph ], [ %inc, %for.body.3 ]
  %mul = mul nsw i32 %x.018, %y.020
  %add = add nsw i32 %x.018, %mul4
  %arrayidx = getelementptr inbounds i32, i32* %a, i32 %add
  store i32 %mul, i32* %arrayidx, align 4
  %inc = add nuw nsw i32 %x.018, 1
  %exitcond = icmp eq i32 %inc, %w
  br i1 %exitcond, label %for.inc.5.loopexit, label %for.body.3

for.inc.5.loopexit:
  br label %for.inc.5

for.inc.5:
  %inc6 = add nuw nsw i32 %y.020, 1
  %exitcond22 = icmp eq i32 %inc6, %h
  br i1 %exitcond22, label %for.end.7.loopexit, label %for.cond.1.preheader

for.end.7.loopexit:
  br label %for.end.7

for.end.7:
  ret void
}

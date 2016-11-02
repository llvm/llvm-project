; RUN: opt -analyze -polly-ast -polly-process-unprofitable < %s | FileCheck %s

; CHECK:      for (int c0 = 0; c0 <= 19999; c0 += 1) {
; CHECK-NEXT:   if (c0 % 2 == 0)
; CHECK-NEXT:     Stmt_if_then(c0);
; CHECK-NEXT:   Stmt_if_end(c0);
; CHECK-NEXT:   if (c0 % 2 == 0)
; CHECK-NEXT:     Stmt_if_then5(c0);
; CHECK-NEXT: }

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n8:16:32-S64"

define void @f(i32* %a, i32 %x) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc
  %i.03 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %rem1 = and i32 %i.03, 1
  %cmp1 = icmp eq i32 %rem1, 0
  br i1 %cmp1, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  %arrayidx = getelementptr inbounds i32, i32* %a, i32 %i.03
  store i32 3, i32* %arrayidx, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  %mul = shl nsw i32 %i.03, 1
  %arrayidx2 = getelementptr inbounds i32, i32* %a, i32 %mul
  store i32 3, i32* %arrayidx2, align 4
  %rem32 = and i32 %i.03, 1
  %cmp4 = icmp eq i32 %rem32, 0
  br i1 %cmp4, label %if.then5, label %for.inc

if.then5:                                         ; preds = %if.end
  %mul6 = mul nsw i32 %i.03, 3
  %arrayidx7 = getelementptr inbounds i32, i32* %a, i32 %mul6
  store i32 3, i32* %arrayidx7, align 4
  br label %for.inc

for.inc:                                          ; preds = %if.end, %if.then5
  %inc = add nsw i32 %i.03, 1
  %cmp = icmp slt i32 %inc, 20000
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.inc
  ret void
}

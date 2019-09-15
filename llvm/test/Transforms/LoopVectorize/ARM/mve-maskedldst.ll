; RUN: opt -loop-vectorize -enable-arm-maskedldst < %s -S -o - | FileCheck %s

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv8.1-m.main-none-eabi"

; CHECK-LABEL: test
; CHECK: llvm.masked.store.v4i32.p0v4i32
define void @test(i32* nocapture %A, i32 %n) #0 {
entry:
  %cmp12 = icmp sgt i32 %n, 0
  br i1 %cmp12, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.inc
  %i.013 = phi i32 [ %inc, %for.inc ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i32 %i.013
  %0 = load i32, i32* %arrayidx, align 4
  %.off = add i32 %0, 9
  %1 = icmp ult i32 %.off, 19
  br i1 %1, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  store i32 0, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %inc = add nuw nsw i32 %i.013, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.inc
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void
}

attributes #0 = { "target-features"="+mve" }

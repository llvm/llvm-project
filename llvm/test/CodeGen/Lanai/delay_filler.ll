; RUN: llc < %s | FileCheck %s
; RUN: llc --lanai-nop-delay-filler < %s | \
; RUN:   FileCheck %s --check-prefix=NOP

; CHECK: bt f
; CHECK-NEXT: or
; NOP: bt f
; NOP-NEXT: nop

; ModuleID = 'delay_filler.c'
target datalayout = "E-m:e-p:32:32-i64:64-a:0:32-n32-S64"
target triple = "lanai"

; Function Attrs: nounwind
define i32 @g(i32 inreg %n) #0 {
entry:
  %cmp5 = icmp sgt i32 %n, 0
  br i1 %cmp5, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  %call.lcssa = phi i32 [ %call, %for.body ]
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  %a.0.lcssa = phi i32 [ undef, %entry ], [ %call.lcssa, %for.cond.cleanup.loopexit ]
  ret i32 %a.0.lcssa

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i.07 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %a.06 = phi i32 [ %call, %for.body ], [ undef, %for.body.preheader ]
  %call = tail call i32 @f(i32 inreg %a.06) #2
  %inc = add nuw nsw i32 %i.07, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body
}

declare i32 @f(i32 inreg) #1


; RUN: llc %s -o - | FileCheck %s
target datalayout = "e-m:o-p:32:32-f64:32:64-f80:128-n8:16:32-S128"
target triple = "i386-apple-macosx10.5"

; Check that the merging of SP updates, when LEAs are involved, happen
; correctly.
; CHECK-LABEL: useLEA:
; CHECK: calll _realloc
; Make sure that the offset we get here is 8 + 16.
; We used to have 8 + 1 because we were not reading the right immediate form
; the LEA instruction.
; CHECK-NEXT: leal 24(%esp), %esp
define noalias ptr @useLEA(ptr nocapture %p, i32 %nbytes) #0 {
entry:
  %cmp = icmp slt i32 %nbytes, 0
  br i1 %cmp, label %cond.end.3, label %cond.false

cond.false:                                       ; preds = %entry
  %tobool = icmp ne i32 %nbytes, 0
  %cond = select i1 %tobool, i32 %nbytes, i32 1
  %call = tail call ptr @realloc(ptr %p, i32 %cond)
  br label %cond.end.3

cond.end.3:                                       ; preds = %entry, %cond.false
  %cond4 = phi ptr [ %call, %cond.false ], [ null, %entry ]
  ret ptr %cond4
}

; Function Attrs: nounwind optsize
declare noalias ptr @realloc(ptr nocapture, i32)

attributes #0 = { nounwind optsize ssp "disable-tail-calls"="false" "frame-pointer"="all" "target-features"="+lea-sp" }

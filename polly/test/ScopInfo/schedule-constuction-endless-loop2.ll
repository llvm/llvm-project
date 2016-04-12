; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
;
; Check that we do not build a SCoP and do not crash.
;
; CHECK-NOT: Statements
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define void @int_upsample(i32* %A) {
entry:
  %0 = load i8, i8* undef, align 1
  %conv7 = zext i8 %0 to i32
  br label %while.body.preheader

while.body.preheader:                             ; preds = %entry
  br label %while.body

while.body:                                       ; preds = %if.end, %while.body.preheader
  %outrow.036 = phi i32 [ %add23, %if.end ], [ 0, %while.body.preheader ]
  br i1 true, label %if.end, label %while.body16

while.body16:                                     ; preds = %while.body16, %while.body
  br label %while.body16.split

while.body16.split:
  br label %while.body16

if.end:                                           ; preds = %while.body
  store i32 0, i32* %A
  %add23 = add nuw nsw i32 %outrow.036, 1
  %cmp = icmp slt i32 %add23, 0
  br i1 %cmp, label %while.body, label %while.end24

while.end24:                                      ; preds = %if.end
  ret void
}

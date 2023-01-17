; RUN: opt %loadPolly -polly-codegen -S < %s | FileCheck %s
;
; Verify we do not crash when we synthezise code for the udiv in the SCoP.
;
; CHECK: polly.start
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define void @RestartModel() #0 {
entry:
  br label %for.cond32.preheader

for.cond32.preheader:                             ; preds = %entry, %for.body50.7
  %i.13 = phi i32 [ 0, %entry ], [ %inc60, %for.body50.7 ]
  %add = add i32 %i.13, 2
  %div44 = udiv i32 undef, %add
  %sub45 = sub i32 16384, %div44
  %conv46 = trunc i32 %sub45 to i16
  br label %for.body35

for.body35:                                       ; preds = %for.cond32.preheader
  br label %for.body50

for.body50:                                       ; preds = %for.body35
  br label %for.body50.1

for.cond62:                                       ; preds = %for.body50.7
  %conv46.lcssa = phi i16 [ %conv46, %for.body50.7 ]
  store i16 %conv46.lcssa, ptr undef, align 2
  br label %for.end83

for.end83:                                        ; preds = %for.cond62
  ret void

for.body50.1:                                     ; preds = %for.body50
  br label %for.body50.2

for.body50.2:                                     ; preds = %for.body50.1
  br label %for.body50.3

for.body50.3:                                     ; preds = %for.body50.2
  br label %for.body50.4

for.body50.4:                                     ; preds = %for.body50.3
  br label %for.body50.5

for.body50.5:                                     ; preds = %for.body50.4
  br label %for.body50.6

for.body50.6:                                     ; preds = %for.body50.5
  br label %for.body50.7

for.body50.7:                                     ; preds = %for.body50.6
  %inc60 = add i32 %i.13, 1
  %cmp29 = icmp ult i32 %inc60, 128
  br i1 %cmp29, label %for.cond32.preheader, label %for.cond62
}

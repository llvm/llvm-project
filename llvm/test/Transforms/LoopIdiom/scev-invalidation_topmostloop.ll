; RUN: opt -S -indvars -loop-idiom -verify -loop-simplifycfg -simplifycfg-require-and-preserve-domtree=1 -loop-idiom < %s | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: @f1()
; CHECK-NEXT: entry:
define void @f1() {
entry:
  br label %lbl1

lbl1:                                             ; preds = %if.end, %entry
  br label %for

for:                                              ; preds = %if.end, %lbl1
  br label %lor.end

lor.end:                                          ; preds = %for
  br i1 undef, label %for.end, label %if.end

if.end:                                           ; preds = %lor.end
  br i1 undef, label %lbl1, label %for

for.end:                                          ; preds = %lor.end
  ret void
}

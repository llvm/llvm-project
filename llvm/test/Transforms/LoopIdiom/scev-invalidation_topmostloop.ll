; RUN: opt -S -passes='loop(indvars,loop-idiom),verify,loop(loop-simplifycfg,loop-idiom)' -simplifycfg-require-and-preserve-domtree=1 < %s | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: @f1(i1 %arg)
; CHECK-NEXT: entry:
define void @f1(i1 %arg) {
entry:
  br label %lbl1

lbl1:                                             ; preds = %if.end, %entry
  br label %for

for:                                              ; preds = %if.end, %lbl1
  br label %lor.end

lor.end:                                          ; preds = %for
  br i1 %arg, label %for.end, label %if.end

if.end:                                           ; preds = %lor.end
  br i1 %arg, label %lbl1, label %for

for.end:                                          ; preds = %lor.end
  ret void
}

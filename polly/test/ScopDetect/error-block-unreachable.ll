; RUN: opt %loadNPMPolly '-passes=print<polly-detect>' -disable-output < %s

; Verify that the scop detection does not crash on inputs with unreachable
; blocks. Earlier we crashed when detecting error blocks.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define void @foo() {
entry:
  br label %while.cond

while.cond:                                       ; preds = %for.end, %entry
  br i1 false, label %for.end, label %while.end8

while.cond1:                                      ; preds = %while.cond4
  br i1 undef, label %while.body3, label %for.inc

while.body3:                                      ; preds = %while.cond1
  br label %while.cond4

while.cond4:                                      ; preds = %while.cond4, %while.body3
  br i1 undef, label %while.cond4, label %while.cond1

for.inc:                                          ; preds = %while.cond1
  %conv = zext i16 undef to i32
  br label %for.end

for.end:                                          ; preds = %for.inc, %while.cond
  %conv.sink = phi i32 [ %conv, %for.inc ], [ 0, %while.cond ]
  br label %while.cond

while.end8:                                       ; preds = %while.cond
  ret void
}

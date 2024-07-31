; RUN: opt %s -passes='print<uniformity>' -disable-output 2>&1 | FileCheck %s

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; CHECK: CYCLES ASSSUMED DIVERGENT:
; CHECK-NEXT: depth=1: entries(if.end16 for.cond1) for.body4

define void @foo(i1 %b) {
entry:
  br i1 %b, label %if.then, label %if.end16

if.then:                                          ; preds = %entry
  br label %for.cond1

for.cond1:                                        ; preds = %if.end16, %for.body4, %if.then
  br i1 false, label %for.body4, label %if.end16

for.body4:                                        ; preds = %for.cond1
  br label %for.cond1

if.end16:                                         ; preds = %for.cond1, %entry
  br label %for.cond1
}

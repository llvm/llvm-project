; RUN: opt -S -passes='loop-mssa(loop-deletion,loop-simplifycfg)' -verify-memoryssa < %s | FileCheck %s
; REQUIRES: asserts

; CHECK-LABEL: @foo()
define void @foo() {
entry:
  br i1 false, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  call void @foo()
  call void @foo()
  br i1 false, label %for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %for.body
  unreachable

for.end:                                          ; preds = %entry
  ret void
}


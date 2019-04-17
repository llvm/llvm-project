; RUN: opt < %s -simplifycfg -S | FileCheck %s

define void @f() {
entry:
  br label %for.cond

for.cond:
  %phi = phi i1 [ false, %entry ], [ true, %for.body ]
  %select = select i1 %phi, i32 1, i32 2
  br label %for.body

for.body:
  switch i32 %select, label %for.cond [
    i32 1, label %return
    i32 2, label %for.body
  ]

return:
  ret void
}

; CHECK-LABEL: define void @f(
; CHECK: br label %[[LABEL:.*]]
; CHECK: br label %[[LABEL]]

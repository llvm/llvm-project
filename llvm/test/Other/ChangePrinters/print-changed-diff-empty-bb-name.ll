; RUN: opt -passes=inline %s -disable-output --print-changed=diff 2>&1 | FileCheck %s

; CHECK: IR Dump After InlinerPass

define void @f(i1 %i) {
  call void @g(i1 %i)
  ret void
}

define void @g(i1 %i) {
  br i1 %i, label %1, label %2

1:
  ret void

2:
  ret void
}

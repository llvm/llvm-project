; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

; CHECK: Function: foo
; CHECK:   MayAlias: i32* %x, i32* %y

define void @foo(ptr noalias %x) {
  %y = call ptr @unclear(ptr %x)
  store i32 0, ptr %x
  store i32 0, ptr %y
  ret void
}

declare ptr @unclear(ptr %a)

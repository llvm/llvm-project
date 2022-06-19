; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

; CHECK: Function: foo
; CHECK:   MayAlias: i32* %x, i32* %y

define void @foo(i32* noalias %x) {
  %y = call i32* @unclear(i32* %x)
  store i32 0, i32* %x
  store i32 0, i32* %y
  ret void
}

declare i32* @unclear(i32* %a)

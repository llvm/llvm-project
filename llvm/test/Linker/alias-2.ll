; RUN: llvm-link %s %S/Inputs/alias-2.ll -S -o - | FileCheck %s
; RUN: llvm-link %S/Inputs/alias-2.ll %s -S -o - | FileCheck %s

; Test the fix for PR26152, where A from the second module is
; erroneously renamed to A.1 and not linked to the declaration from
; the first module

@C = alias void (), ptr @A

define void @D() {
  call void @C()
  ret void
}

define void @A() {
  ret void
}

; CHECK-DAG: @C = alias void (), ptr @A
; CHECK-DAG: define void @B()
; CHECK-DAG:   call void @A()
; CHECK-DAG: define void @D()
; CHECK-DAG:   call void @C()
; CHECK-DAG: define void @A()

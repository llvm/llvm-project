; Check that `getRegForInlineAsmConstraint` does not crash on empty Constraint.
; RUN: llc -mtriple=mips64 < %s | FileCheck %s

define void @foo() {
entry:
  %s = alloca i32, align 4
  %x = alloca i32, align 4
  call void asm "", "=*imr,=*m,0,*m,~{$1}"(ptr elementtype(i32) %x, ptr elementtype(i32) %s, ptr %x, ptr elementtype(i32) %s)

; CHECK: #APP
; CHECK: #NO_APP

  ret void
}

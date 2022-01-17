; Check that `getRegForInlineAsmConstraint` does not crash on empty Constraint.
; RUN: llc -march=mips64 < %s | FileCheck %s

define void @foo() {
entry:
  %s = alloca i32, align 4
  %x = alloca i32, align 4
  call void asm "", "=*imr,=*m,0,*m,~{$1}"(i32* elementtype(i32) %x, i32* elementtype(i32) %s, i32* %x, i32* elementtype(i32) %s)

; CHECK: #APP
; CHECK: #NO_APP

  ret void
}

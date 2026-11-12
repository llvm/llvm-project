; RUN: llc -mtriple=mips < %s | FileCheck %s

; Check for the emission of appropriate assembler directives before and
; after the inline assembly code.
define void @f() nounwind {
entry:
; CHECK:      #APP
; CHECK-NEXT: .set  push
; CHECK-NEXT: .set  at
; CHECK-NEXT: .set  macro
; CHECK-NEXT: .set  reorder
; CHECK:      addi $9, ${{[2-9][0-9]?}}, 8
; CHECK:      ori ${{[2-9][0-9]?}}, $9, 6
; CHECK:      .set  pop
; CHECK-NEXT: #NO_APP
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  store i32 20, ptr %a, align 4
  %0 = load i32, ptr %a, align 4
  %1 = call i32 asm sideeffect "addi $$9, $1, 8\0A\09ori $0, $$9, 6", "=r,r,~{$1}"(i32 %0)
  store i32 %1, ptr %b, align 4
  ret void
}

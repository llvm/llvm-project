; RUN: llc -mtriple=mips < %s | FileCheck %s

@a = global i8 0, align 1

define void @b() {
entry:
  %0 = load i8, ptr @a, align 1
  %tobool = trunc i8 %0 to i1
  call void asm sideeffect "", "Jr,~{$1}"(i1 %tobool)
  ret void
}

; CHECK:      lui $1, %hi(a)
; CHECK-NEXT: lbu $2, %lo(a)($1)

; RUN: llc -mtriple=thumb-eabi -no-integrated-as %s -o - | FileCheck %s

define i32 @t1(i32 %x, i32 %y) nounwind {
entry:
  ; CHECK: mov r0, r12
  %0 = tail call i32 asm "mov $0, $1", "=l,h"(i32 %y) nounwind
  ret i32 %0
}

; CHECK-LABEL: constraint_r:
; CHECK: foo2 r{{[0-7]+}}, r{{[0-7]+}}

define i32 @constraint_r() {
entry:
  %0 = tail call i32 asm sideeffect "movs $0, #1", "=r"()
  tail call void asm sideeffect "foo1", "~{r0},~{r1},~{r2},~{r3},~{r4},~{r5},~{r6},~{r7}"()
  %1 = tail call i32 asm sideeffect "foo2 $0, $1", "=r,r"(i32 %0)
  ret i32 %1
}

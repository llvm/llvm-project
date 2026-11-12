; RUN: llc < %s | FileCheck %s

; Test that Lanai select instruction is selected from LLVM select instruction.

target datalayout = "E-m:e-p:32:32-i64:64-a:0:32-n32-S64"
target triple = "lanai"

; CHECK-LABEL: select_i32_bool:
; CHECK: sub.f %r6, 0x0, %r0
; CHECK: sel.ne %r7, %r18, %rv
define i32 @select_i32_bool(i1 zeroext inreg %a, i32 inreg %b, i32 inreg %c) {
  %cond = select i1 %a, i32 %b, i32 %c
  ret i32 %cond
}

; CHECK-LABEL: select_i32_eq:
; CHECK: sub.f %r6, 0x0, %r0
; CHECK: sel.eq %r7, %r18, %rv
define i32 @select_i32_eq(i32 inreg %a, i32 inreg %b, i32 inreg %c) {
  %cmp = icmp eq i32 %a, 0
  %cond = select i1 %cmp, i32 %b, i32 %c
  ret i32 %cond
}

; CHECK-LABEL: select_i32_ne:
; CHECK: sub.f %r6, 0x0, %r0
; CHECK: sel.ne %r7, %r18, %rv
define i32 @select_i32_ne(i32 inreg %a, i32 inreg %b, i32 inreg %c) {
  %cmp = icmp ne i32 %a, 0
  %cond = select i1 %cmp, i32 %b, i32 %c
  ret i32 %cond
}

; CHECK-LABEL: select_i32_lt:
; CHECK: sub.f %r6, %r7, %r0
; CHECK: sel.lt %r6, %r7, %rv
define i32 @select_i32_lt(i32 inreg %x, i32 inreg %y) #0 {
  %1 = icmp slt i32 %x, %y
  %2 = select i1 %1, i32 %x, i32 %y
  ret i32 %2
}

; CHECK-LABEL: select_add_setcc:
; CHECK: sub.f %r6, 0x0, %r0
; CHECK: add %r7, 0x7, [[REG:%r[0-9]+]]
; CHECK: sel.eq %r7, [[REG]], %rv
define i32 @select_add_setcc(i32 inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %a, 0
  %sel = select i1 %cmp, i32 0, i32 7
  %sum = add i32 %b, %sel
  ret i32 %sum
}

; CHECK-LABEL: select_binop_non_setcc_no_hang:
; CHECK: sel.
; CHECK: jmp %rca
define i32 @select_binop_non_setcc_no_hang(i32 inreg %a, i32 inreg %b) {
  %c1 = icmp sgt i32 %a, 0
  %c2 = icmp slt i32 %b, 8
  %and = and i1 %c1, %c2
  %s1 = select i1 %and, i32 0, i32 7
  %d1 = sdiv i32 %a, 8
  %d2 = sdiv i32 %b, 8
  %sum1 = add i32 %d1, %s1
  %sum2 = add i32 %sum1, %d2
  %sum3 = add i32 %sum2, 4
  ret i32 %sum3
}

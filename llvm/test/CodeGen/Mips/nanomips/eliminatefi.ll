; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s

define void @move_sp(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: move_sp
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  %c.addr = alloca i32, align 4
; CHECK: addiu $a0, $sp, 8
; CHECK: addiu $a1, $sp, 4
; CHECK: move $a2, $sp
; CHECK-NOT: addiu $a1, $sp, 0
  store i32 %a, i32* %a.addr, align 4
  store i32 %b, i32* %b.addr, align 4
  store i32 %c, i32* %c.addr, align 4
  call void @bar(i32* %a.addr, i32* %b.addr, i32* %c.addr)
  ret void
}

declare void @bar(i32*, i32*, i32*)

; RUN: llc %s -o - | FileCheck %s

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc18.0.31101"

define i32 @_xbegin() {
entry:
  %res = alloca i32, align 4
  store i32 -1, ptr %res, align 4
  call void asm sideeffect inteldialect ".byte 0xC7\0A\09.byte 0xF8\0A\09.byte 2\0A\09.byte 0\0A\09.byte 0\0A\09.byte 0\0A\09jmp .L__MSASMLABEL_.0__L2\0A\09mov dword ptr $0, eax\0A\09.L__MSASMLABEL_.0__L2:", "=*m,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i32) nonnull %res)
  %0 = load i32, ptr %res, align 4
  ret i32 %0
}

; CHECK-NOT: Error parsing inline asm

; CHECK-LABEL: _xbegin:
; CHECK: jmp .L__MSASMLABEL_.0__L2
; CHECK: .L__MSASMLABEL_.0__L2:

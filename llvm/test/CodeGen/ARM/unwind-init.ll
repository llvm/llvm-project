; RUN: llc -mtriple=armv7-unknown-linux-gnueabi < %s | FileCheck %s
; Check that all callee-saved registers are saved and restored in functions
; that call __builtin_unwind_init(). This is its undocumented behavior in gcc,
; and it is used in compiling libgcc_eh.
; See also PR8541

declare void @llvm.eh.unwind.init()

define void @calls_unwind_init() {
  call void @llvm.eh.unwind.init()
  ret void
}

; CHECK: calls_unwind_init:
; CHECK: push    {r4, r5, r6, r7, r8, r9, r10, r11, lr}
; CHECK: vpush   {d8, d9, d10, d11, d12, d13, d14, d15}
; CHECK: vpop    {d8, d9, d10, d11, d12, d13, d14, d15}
; CHECK: pop     {r4, r5, r6, r7, r8, r9, r10, r11, pc}

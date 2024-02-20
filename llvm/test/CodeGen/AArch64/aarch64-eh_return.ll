; RUN: llc < %s -O0 -mtriple aarch64-none-linux-gnu | FileCheck %s

define  void @eh1(i32 noundef %x, ptr noundef %p) nounwind {
; CHECK-LABEL: eh1:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    stp x29, x30, [sp, #-16]!
; CHECK-NEXT:    mov x29, sp
; CHECK-NEXT:    sub sp, sp, #48
; CHECK-NEXT:    stp x0, x1, [sp]
; CHECK-NEXT:    stp x2, x3, [sp, #16]
; CHECK-NEXT:    stur w0, [x29, #-4]
; CHECK-NEXT:    stur x1, [x29, #-16]
; CHECK-NEXT:    ldursw x4, [x29, #-4]
; CHECK-NEXT:    ldur x0, [x29, #-16]
; CHECK-NEXT:    stur x0, [x29, #6]
; CHECK-NEXT:    add sp, sp, #48
; CHECK-NEXT:    ldp x0, x1, [x29, #-48]
; CHECK-NEXT:    ldp x2, x3, [x29, #-32]
; CHECK-NEXT:    ldp x29, x30, [sp], #16
; CHECK-NEXT:    add sp, sp, x4, uxtb
; CHECK-NEXT:    ret
entry:
  %x.addr = alloca i32, align 4
  %p.addr = alloca ptr, align 8
  store i32 %x, ptr %x.addr, align 4
  store ptr %p, ptr %p.addr, align 8
  %0 = load i32, ptr %x.addr, align 4
  %conv = sext i32 %0 to i64
  %1 = load ptr, ptr %p.addr, align 8
  call void @llvm.eh.return.i64(i64 %conv, ptr %1)
  unreachable
}

declare void @llvm.eh.return.i64(i64, ptr)


; RUN: llc -mtriple=xtensa -mcpu=esp32s3 -O2 -verify-machineinstrs < %s \
; RUN:   | FileCheck %s

; Windowed ABI: any SP change after ENTRY must go through MOVSP so the
; register-window save area is relocated (a plain ADD corrupts the caller
; if a window spill lands between ENTRY and the update). Over-aligned
; frames therefore realign via scratch ADD + MOVSP, computed off SP.

declare void @use(ptr)

define void @realign64() {
; CHECK-LABEL: realign64:
; CHECK: entry a1, 160
; CHECK-NOT: add{{(\.n)?}} a1, a1
; CHECK: and a8, a1, a8
; CHECK-NOT: add{{(\.n)?}} a1, a1
; CHECK: movsp a1, a8
; CHECK-NOT: add{{(\.n)?}} a1, a1
; CHECK: retw
entry:
  %buf = alloca [64 x i8], align 64
  call void @use(ptr %buf)
  ret void
}

; With a frame pointer, FP (a7) still holds an incoming argument when the
; misalignment is computed, so the AND must be sourced from SP (a1), and
; FP is only materialized after the realignment.
define void @realign64_fp(i32 %x0, i32 %x1, i32 %x2, i32 %x3, i32 %x4, i32 %x5, i32 %n) {
; CHECK-LABEL: realign64_fp:
; CHECK: entry a1, 160
; CHECK-NOT: add{{(\.n)?}} a1, a1
; CHECK: and a8, a1, a8
; CHECK-NOT: add{{(\.n)?}} a1, a1
; CHECK: movsp a1, a8
; CHECK: or a7, a1, a1
; CHECK-NOT: add{{(\.n)?}} a1, a1
; CHECK: retw
entry:
  %buf = alloca [64 x i8], align 64
  %dyn = alloca i8, i32 %n
  call void @use(ptr %buf)
  call void @use(ptr %dyn)
  ret void
}

; SP is only 16-byte aligned, so align 32 needs realignment too.
define void @realign32() {
; CHECK-LABEL: realign32:
; CHECK: entry a1, 64
; CHECK-NOT: add{{(\.n)?}} a1, a1
; CHECK: and a8, a1, a8
; CHECK-NOT: add{{(\.n)?}} a1, a1
; CHECK: movsp a1, a8
; CHECK-NOT: add{{(\.n)?}} a1, a1
; CHECK: retw
entry:
  %buf = alloca [32 x i8], align 32
  call void @use(ptr %buf)
  ret void
}

; Align 16 needs no realignment: SP is already 16-byte aligned.
define void @realign16() {
; CHECK-LABEL: realign16:
; CHECK: entry a1, 64
; CHECK-NOT: movsp
; CHECK: retw
entry:
  %buf = alloca [32 x i8], align 16
  call void @use(ptr %buf)
  ret void
}

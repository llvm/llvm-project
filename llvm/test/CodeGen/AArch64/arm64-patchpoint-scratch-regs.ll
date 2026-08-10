; RUN: llc -mtriple=arm64-apple-darwin -enable-misched=0 -mcpu=cyclone < %s | FileCheck %s

; Test that scratch registers are spilled around patchpoints
; CHECK: InlineAsm End
; CHECK-NEXT: mov x{{[0-9]+}}, x16
; CHECK-NEXT: mov x{{[0-9]+}}, x17
; CHECK-NEXT: Ltmp
; CHECK-NEXT: nop
define void @clobberScratch(ptr %p) {
  %v = load i32, ptr %p
  tail call void asm sideeffect "nop", "~{x0},~{x1},~{x2},~{x3},~{x4},~{x5},~{x6},~{x7},~{x8},~{x9},~{x10},~{x11},~{x12},~{x13},~{x14},~{x15},~{x18},~{x19},~{x20},~{x21},~{x22},~{x23},~{x24},~{x25},~{x26},~{x27},~{x28},~{x29},~{x30},~{x31}"() nounwind
  tail call void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 5, i32 20, ptr null, i32 0, ptr %p, i32 %v)
  store i32 %v, ptr %p
  ret void
}

declare void @llvm.experimental.patchpoint.void(i64, i32, ptr, i32, ...)


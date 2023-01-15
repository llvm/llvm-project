; RUN: llc -mtriple=arm64-apple-macosx10.8.0 < %s -verify-coalescing
; RUN: llc -mtriple=arm64_32-apple-ios9.0 < %s -verify-coalescing
; <rdar://problem/11522048>

; Verify that we can handle spilling the stack pointer without attempting
; spilling it directly.
; CHECK: f
; CHECK: mov [[X0:x[0-9]+]], sp
; CHECK: str [[X0]]
; CHECK: inlineasm
define void @f() nounwind ssp {
entry:
  %savedstack = call ptr @llvm.stacksave() nounwind
  call void asm sideeffect "; inlineasm", "~{x0},~{x1},~{x2},~{x3},~{x4},~{x5},~{x6},~{x7},~{x8},~{x9},~{x10},~{x11},~{x12},~{x13},~{x14},~{x15},~{x16},~{x17},~{x18},~{x19},~{x20},~{x21},~{x22},~{x23},~{x24},~{x25},~{x26},~{x27},~{x28},~{fp},~{lr},~{sp},~{memory}"() nounwind
  call void @llvm.stackrestore(ptr %savedstack) nounwind
  ret void
}

declare ptr @llvm.stacksave() nounwind
declare void @llvm.stackrestore(ptr) nounwind

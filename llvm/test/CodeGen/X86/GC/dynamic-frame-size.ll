; RUN: llc < %s | FileCheck %s
target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

declare void @use(ptr)

; Test that a frame which requires dynamic relocation produces a stack map
; with a size of UINT64_MAX.
define void @test(ptr %ptr) gc "erlang" {
   ; 32 byte alignment (for the alloca) is larger than the default
   ; 16 byte alignment
   %slot = alloca <4 x ptr>
   call void @use(ptr %slot);
   ret void
}

; CHECK: .note.gc
; CHECK-NEXT: .p2align 3
; safe point count
; CHECK: .short	1
; CHECK: .long	.Ltmp0
; stack frame size (in words)
; CHECK: .short	-1
; stack arity (arguments on the stack)
; CHECK: .short	0
; live root count
; CHECK: .short	0


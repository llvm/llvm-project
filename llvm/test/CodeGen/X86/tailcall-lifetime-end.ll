; RUN: llc -mtriple=x86_64-unknown-linux-gnu -o - %s | FileCheck %s

; A lifetime end intrinsic should not prevent a call from being tail call
; optimized.

define void @foobar() {
; CHECK-LABEL: foobar
; CHECK: pushq	%rax
; CHECK: leaq	4(%rsp), %rdi
; CHECK: callq	foo
; CHECK: popq	%rax
; CHECK: jmp	bar
entry:
  %i = alloca i32
  %0 = bitcast i32* %i to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %0)
  call void @foo(i32* nonnull %i)
  tail call void @bar()
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %0)
  ret void
}

declare void @foo(i32* nocapture %p)
declare void @bar()

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)

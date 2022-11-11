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
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %i)
  call void @foo(ptr nonnull %i)
  tail call void @bar()
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %i)
  ret void
}

declare void @foo(ptr nocapture %p)
declare void @bar()

declare void @llvm.lifetime.start.p0(i64, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64, ptr nocapture)

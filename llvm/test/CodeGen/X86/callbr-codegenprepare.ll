;; RUN: opt -S -codegenprepare < %s | FileCheck %s

;; Ensure that codegenprepare (via InstSimplify) doesn't eliminate the
;; phi here (which would cause a module verification error).

;; CHECK: phi

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @foo(i32)

define dso_local i32 @futex_lock_pi_atomic() local_unnamed_addr {
entry:
  %0 = callbr i32 asm "", "=r,i,~{dirflag},~{fpsr},~{flags}"(i8* blockaddress(@futex_lock_pi_atomic, %b.exit))
          to label %asm.fallthrough.i [label %b.exit]

asm.fallthrough.i:
  br label %b.exit

b.exit:
  %g.0 = phi i32 [ %0, %asm.fallthrough.i ], [ undef, %entry ]
  tail call void @foo(i32 %g.0)
  ret i32 undef
}


; RUN: llc -verify-machineinstrs < %s
; Regression test for https://github.com/llvm/llvm-project/issues/58911

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7-none-unknown-eabi"

@a = dso_local global i64 0, align 8
@d = dso_local local_unnamed_addr global i32 0, align 4

define dso_local void @f() nounwind {
entry:
  store volatile i64 0, ptr @a, align 8
  %0 = load i32, ptr @d, align 4
  %tobool.not = icmp eq i32 %0, 0
  %conv = zext i32 %0 to i64
  %sub = sub nsw i64 0, %conv
  %cond = select i1 %tobool.not, i64 0, i64 %sub
  store volatile i64 %cond, ptr @a, align 8
  ret void
}


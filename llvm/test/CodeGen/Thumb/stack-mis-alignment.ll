; RUN: llc -O0 < %s | FileCheck %s

; For noreturn function with StackAlignment 8 (function contains call/alloc),
; check that lr is saved to keep the stack aligned.
; CHECK: push    {lr}

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv5e-none-linux-gnueabi"

define dso_local i32 @f() noreturn nounwind {
entry:
  call i32 @llvm.arm.space(i32 2048, i32 undef)
  tail call i32 @exit(i32 0)
  unreachable
}

declare i32 @llvm.arm.space(i32, i32)
declare dso_local i32 @exit(i32)

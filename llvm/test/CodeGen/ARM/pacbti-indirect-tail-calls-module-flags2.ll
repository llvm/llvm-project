; RUN: llc -mtriple=thumbv8.1m.main-none-none-eabi -mattr=+pacbti< %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv8.1m.main-m.main-unknown"

define dso_local void @sgign_return_address_all(ptr noundef readonly %fptr_arg) local_unnamed_addr #0 {
entry:
  %0 = tail call ptr asm "", "={r12},{r12},~{lr}"(ptr %fptr_arg)
  tail call void %0()
; CHECK: bx {{r0|r1|r2|r3}}
  ret void
}

!llvm.module.flags = !{!1}

!1 = !{i32 8, !"sign-return-address", i32 1}
!2 = !{i32 8, !"sign-return-address-all", i32 1}


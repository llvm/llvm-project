; RUN: llc < %s -mtriple=armv7a-arm-none-eabi | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7-arm-none-eabi"

define void @foo() minsize {
entry:
  ; CHECK: .vsave	{[[SAVE_REG:d[0-9]+]]}
  ; CHECK-NEXT: .pad #8
  ; CHECK-NEXT: vpush   {[[PAD_REG:d[0-9]+]], [[SAVE_REG]]}
  ; CHECK: vpop     {[[PAD_REG]], [[SAVE_REG]]}
  %a = alloca i32, align 4
  call void asm sideeffect "", "r,~{d8}"(i32* %a)
  ret void
}

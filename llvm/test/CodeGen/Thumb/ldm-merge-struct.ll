; RUN: llc -mtriple=thumbv6m-eabi -verify-machineinstrs %s -o - | FileCheck %s
target datalayout = "e-m:e-p:32:32-i1:8:32-i8:8:32-i16:16:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv6m-none--eabi"

%struct.S = type { i32, i32 }

@s = common global %struct.S zeroinitializer, align 4

define i32 @f() {
entry:
; CHECK-LABEL: f:
; CHECK: ldm r[[BASE:[0-9]]],
; CHECK-NOT: subs r[[BASE]]
  %0 = load i32, i32* getelementptr inbounds (%struct.S, %struct.S* @s, i32 0, i32 0), align 4
  %1 = load i32, i32* getelementptr inbounds (%struct.S, %struct.S* @s, i32 0, i32 1), align 4
  %cmp = icmp sgt i32 %0, %1
  %2 = sub i32 0, %1
  %cond.p = select i1 %cmp, i32 %1, i32 %2
  %cond = add i32 %cond.p, %0
  ret i32 %cond
}

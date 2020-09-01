; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv6m-arm-none-eabi"

; CHECK-LABEL: f:
; CHECK: bl g
; CHECK: blx r
; CHECK: bl g
; CHECK: bl g
define void @f(i32* %p, i32 %x, i32 %y, i32 %z) minsize optsize {
entry:
  call void @g(i32* %p, i32 %x, i32 %y, i32 %z)
  call void @g(i32* %p, i32 %x, i32 %y, i32 %z)
  call void @g(i32* %p, i32 %x, i32 %y, i32 %z)
  call void @g(i32* %p, i32 %x, i32 %y, i32 %z)
  ret void
}

declare void @g(i32*,i32,i32,i32)

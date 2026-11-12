; RUN: opt -S < %s -unreachableblockelim | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @abort()

; CHECK-LABEL: @foo(
; CHECK-NOT: return:
define void @foo(ptr %p) {
entry:
  %p.addr = alloca ptr, align 8
  call void @abort()
  unreachable

return:                                           ; No predecessors!
  store ptr %p, ptr %p.addr, align 8
  ret void
}


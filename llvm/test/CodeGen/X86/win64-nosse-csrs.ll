; RUN: llc < %s -mattr="-sse,-mmx,+soft-float" | FileCheck %s

; CHECK: peach:
; CHECK: pushq %rsi
; CHECK: pushq %rdi
; CHECK-NOT: movaps
; CHECK: callq banana
; CHECK-NOT: movaps
; CHECK: popq %rdi
; CHECK: popq %rsi
; CHECK: retq

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: uwtable
define internal i64 @banana() unnamed_addr #0 {
entry-block:
  ret i64 0
}

; Function Attrs: nounwind uwtable
define x86_64_win64cc i64 @peach() unnamed_addr #1 {
entry-block:
  %0 = call i64 @banana()
  ret i64 %0
}

attributes #0 = { uwtable }
attributes #1 = { nounwind uwtable }

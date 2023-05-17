; RUN: llc -O1 -regalloc=greedy -mtriple=x86_64-apple-macosx < %s -o - | FileCheck %s
; Check that last chance split (RAGreedy::tryInstructonSplit) just split
; when this is beneficial, otherwise we end up with uncoalesced copies.
; <rdar://problem/15570057> 

target datalayout = "e-i64:64-f80:128-s:64-n8:16:32:64-S128"

@f = external constant ptr

; CHECK-LABEL: test:
; Get the address of f in the GOT.
; CHECK: movq _f@{{[^,]+}}, [[F_ENTRY_ADDR:%[a-z0-9]+]]
; Read the actual address of f.
; CHECK: movq ([[F_ENTRY_ADDR]]), [[F_ADDR:%[a-z0-9]+]]
; Check that we do not have useless split points before each call.
; CHECK-NOT: movq
; CHECK: callq *[[F_ADDR]]
; Check that we do not have useless split points before each call.
; CHECK-NOT: movq
; CHECK: callq *[[F_ADDR]]
; Last call is a tail call, thus the address of the function cannot use
; a callee saved register.
; CHECK: movq [[F_ADDR]], [[F_ADDR_TC:%[a-z0-9]+]]
; CHECK: popq [[F_ADDR]]
; CHECK: jmpq *[[F_ADDR_TC]]
define void @test(i32 %a, i32 %b, i32 %c) {
entry:
  %fct_f = load ptr, ptr @f, align 8
  tail call void %fct_f(i32 %a)
  tail call void %fct_f(i32 %b)
  tail call void %fct_f(i32 %c)
  ret void
}

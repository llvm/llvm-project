; REQUIRES: asserts
; RUN: opt -passes=loop-reduce -lsr-drop-solution -debug-only=loop-reduce -disable-output %s 2>&1 | FileCheck %s --implicit-check-not='Baseline is more profitable'

target datalayout = "e-m:e-p:64:64-i64:64-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; These fixups share one formula. Charge every offset, sharing register costs.
define void @multiple_fixups(ptr %base, i64 %n) {
; CHECK-LABEL: LSR on loop %offset.loop:
; CHECK: The baseline solution requires 2 instructions 2 regs, with addrec cost 2, plus 9 imm cost, plus 2 setup cost
; CHECK: The chosen solution requires 2 instructions 2 regs, with addrec cost 2, plus 7 imm cost, plus 3 setup cost:
entry:
  br label %offset.loop
offset.loop:
  %i = phi i64 [ 0, %entry ], [ %next, %offset.loop ]
  %p0 = getelementptr i32, ptr %base, i64 %i
  %p1 = getelementptr i32, ptr %p0, i64 1
  %p2 = getelementptr i32, ptr %p0, i64 2
  store volatile i32 1, ptr %p0, align 4
  store volatile i32 1, ptr %p1, align 4
  store volatile i32 1, ptr %p2, align 4
  %next = add nuw i64 %i, 1
  %done = icmp eq i64 %next, %n
  br i1 %done, label %exit, label %offset.loop
exit:
  ret void
}

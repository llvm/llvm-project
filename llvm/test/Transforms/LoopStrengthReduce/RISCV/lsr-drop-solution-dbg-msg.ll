; REQUIRES: asserts
; RUN: opt -passes=loop-reduce -lsr-drop-solution=true -debug-only=loop-reduce -disable-output %s 2>&1 | FileCheck %s --check-prefixes=COST,DROP
; RUN: opt -passes=loop-reduce -lsr-drop-solution=false -debug-only=loop-reduce -disable-output %s 2>&1 | FileCheck %s --check-prefixes=COST,KEEP
; RUN: opt -passes=loop-reduce -debug-only=loop-reduce -disable-output %s 2>&1 | FileCheck %s --check-prefixes=COST,KEEP

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "riscv64-unknown-linux-gnu"

; Sharing the byte-offset IV saves registers but costs an extra instruction.
define void @reject_offset_reuse(ptr %base) {
; COST-LABEL: LSR on loop %drop.loop:
; COST: The baseline solution requires 4 instructions 4 regs, with addrec cost 4, plus 4 setup cost
; COST: The chosen solution requires 5 instructions 2 regs, with addrec cost 1, plus 3 base adds, plus 11 imm cost, plus 1 setup cost:
; DROP: Baseline is more profitable than chosen solution, dropping LSR solution.
; KEEP: Baseline is more profitable than chosen solution, add option 'lsr-drop-solution' to drop LSR solution.
entry:
  br label %drop.loop

drop.loop:
  %offset = phi i64 [ 0, %entry ], [ %next, %drop.loop ]
  %p0 = getelementptr i8, ptr %base, i64 %offset
  %p1 = getelementptr i8, ptr %p0, i64 16
  call void @consume(i64 %offset, ptr %p0, ptr %p1)
  %next = add nuw i64 %offset, 4
  %done = icmp eq i64 %next, 1024
  br i1 %done, label %exit, label %drop.loop

exit:
  ret void
}

declare void @consume(i64, ptr, ptr)

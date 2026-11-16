; REQUIRES: asserts
; RUN: llc -O2 -stop-after=loop-reduce -lsr-drop-solution -debug-only=loop-reduce -o /dev/null %s 2>&1 | FileCheck %s --implicit-check-not='Baseline is more profitable'

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "riscv64-unknown-linux-gnu"

; The supplemental use pins the base already counted by the address formula.
define void @supplemental_base(ptr %base, i64 %n) {
; CHECK-LABEL: LSR on loop %supplemental.loop:
; CHECK: The baseline solution requires 2 instructions 3 regs, with addrec cost 2, plus 3 setup cost
; CHECK: The chosen solution requires 2 instructions 3 regs, with addrec cost 2, plus 3 setup cost:
entry:
  br label %supplemental.loop
supplemental.loop:
  %i = phi i64 [ 0, %entry ], [ %next, %supplemental.loop ]
  %p = getelementptr i32, ptr %base, i64 %i
  store i32 1, ptr %p, align 4
  %next = add nuw i64 %i, 1
  %done = icmp eq i64 %next, %n
  br i1 %done, label %exit, label %supplemental.loop
exit:
  call void @use_ptr(ptr %base)
  ret void
}

; Price the original live-out fixup too. The CodeGen pipeline leaves it fully
; outside the loop, whereas opt's loop adaptor would first form an LCSSA PHI.
define i64 @iv_live_out(ptr %base, i64 %n, i64 %stride) {
; CHECK-LABEL: LSR on loop %liveout.loop:
; CHECK: The baseline solution requires 3 instructions 5 regs, with addrec cost 3, plus 5 setup cost
; CHECK: The chosen solution requires 3 instructions 5 regs, with addrec cost 3, plus 5 setup cost:
entry:
  br label %liveout.loop
liveout.loop:
  %i = phi i64 [ 0, %entry ], [ %next, %liveout.loop ]
  %offset = phi i64 [ 0, %entry ], [ %offset.next, %liveout.loop ]
  %p = getelementptr i8, ptr %base, i64 %offset
  store i8 1, ptr %p, align 1
  %offset.next = add i64 %offset, %stride
  %liveout = add i64 %offset.next, 262144
  %next = add nuw i64 %i, 1
  %done = icmp eq i64 %next, %n
  br i1 %done, label %exit, label %liveout.loop
exit:
  call void @use_i64(i64 %liveout)
  ret i64 %liveout
}

declare void @use_ptr(ptr)
declare void @use_i64(i64)

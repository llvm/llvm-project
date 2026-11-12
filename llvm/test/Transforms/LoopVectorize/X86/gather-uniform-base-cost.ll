; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -mcpu=skylake-avx512 -force-vector-width=16 \
; RUN:   -force-vector-interleave=1 -debug-only=loop-vectorize -disable-output \
; RUN:   %s 2>&1 | FileCheck %s

; A gather cost has to come out the same in both cost models. The scalar
; address does not say which of its operands varies across lanes, so each model
; answers that question for the target. Here the base is loaded before the
; loop, so it is uniform but is still an instruction. A model that stays quiet
; leaves the target to assume the base varies, which keeps pointer-width
; indices and charges two parts for one gather.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: LV: Memory widening: InterleaveCost = Invalid, GatherScatterCost = 18
; CHECK: Cost of 18 for VF 16: WIDEN ir<%v> = load ir<%p>, ir<%c>, align 4

define void @gather_uniform_instr_base(ptr %pp, ptr noalias %out, ptr noalias %trigger, ptr noalias %index) {
entry:
  %base = load ptr, ptr %pp, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  %gt = getelementptr inbounds i32, ptr %trigger, i64 %i
  %t = load i32, ptr %gt, align 4
  %c = icmp sgt i32 %t, 0
  br i1 %c, label %if.then, label %latch

if.then:
  %gi = getelementptr inbounds i32, ptr %index, i64 %i
  %idx32 = load i32, ptr %gi, align 4
  %idx = sext i32 %idx32 to i64
  %p = getelementptr inbounds float, ptr %base, i64 %idx
  %v = load float, ptr %p, align 4
  %add = fadd float %v, 5.000000e-01
  %go = getelementptr inbounds float, ptr %out, i64 %i
  store float %add, ptr %go, align 4
  br label %latch

latch:
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, 4096
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; RUN: llc < %s -march=xcore -asm-verbose=0 | FileCheck %s

; MachineLICM should hoist the LDWCP out of the loop.

; CHECK: f:
; CHECK-NEXT: ldw [[REG:r[0-9]+]], cp[.LCPI0_0]
; CHECK-NEXT: .LBB0_1:
; CHECK-NEXT: stw [[REG]], r0[0]
; CHECK-NEXT: bu .LBB0_1

define void @f(i32* nocapture %p) noreturn nounwind {
entry:
  br label %bb

bb:                                               ; preds = %bb, %entry
  store volatile i32 525509670, i32* %p, align 4
  br label %bb
}

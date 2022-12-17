; RUN: opt -S < %s -passes=loop-unroll -unroll-count=4 | FileCheck %s
;
; Test induction variable simplify after loop unrolling. It should
; expose nice opportunities for GVN.
;
; CHECK-NOT: while.body also ensures that loop unrolling (with SCEV)
; removes unrolled loop exits given that 128 is a multiple of 4.

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f80:128:128-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"

; PR10534: LoopUnroll not keeping canonical induction variable...
; CHECK: while.body:
; CHECK-NOT: while.body.1:
; CHECK: %shr.1 = lshr i32 %bit_addr.addr.01, 5
; CHECK: %arrayidx.1 = getelementptr inbounds i32, i32* %bitmap, i32 %shr.1
; CHECK: %shr.2 = lshr i32 %bit_addr.addr.01, 5
; CHECK: %arrayidx.2 = getelementptr inbounds i32, i32* %bitmap, i32 %shr.2
; CHECK: %shr.3 = lshr i32 %bit_addr.addr.01, 5
; CHECK: %arrayidx.3 = getelementptr inbounds i32, i32* %bitmap, i32 %shr.3
define void @FlipBit(i32* nocapture %bitmap, i32 %bit_addr, i32 %nbits) nounwind {
entry:
  br label %while.body

while.body:
  %nbits.addr.02 = phi i32 [ 128, %entry ], [ %dec, %while.body ]
  %bit_addr.addr.01 = phi i32 [ 0, %entry ], [ %inc, %while.body ]
  %dec = add i32 %nbits.addr.02, -1
  %shr = lshr i32 %bit_addr.addr.01, 5
  %rem = and i32 %bit_addr.addr.01, 31
  %shl = shl i32 1, %rem
  %arrayidx = getelementptr inbounds i32, i32* %bitmap, i32 %shr
  %tmp6 = load i32, i32* %arrayidx, align 4
  %xor = xor i32 %tmp6, %shl
  store i32 %xor, i32* %arrayidx, align 4
  %inc = add i32 %bit_addr.addr.01, 1
  %tobool = icmp eq i32 %dec, 0
  br i1 %tobool, label %while.end, label %while.body

while.end:
  ret void
}

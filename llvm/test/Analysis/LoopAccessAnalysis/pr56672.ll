; RUN: opt -passes='loop(loop-rotate),print-access-info' -S %s 2>&1 | FileCheck %s
; RUN: opt -passes='loop(loop-rotate),invalidate<loops>,print-access-info' -S %s 2>&1 | FileCheck %s

; Make sure that the result of analysis is consistent regardless of blocks
; order as they are stored in loop. This test demonstrates the situation when
; recomputation of LI produces loop with different blocks order, and LA gives
; a different result for it. The reason of this bug hasn't been found yet, but
; the algorithm is somehow dependent on blocks order.
define void @test_01(i32* %p) {
; CHECK-LABEL: test_01
; CHECK:       Report: unsafe dependent memory operations in loop.
; CHECK-NOT:   Memory dependences are safe
entry:
  br label %loop

loop.progress:                                              ; preds = %loop
  br label %loop.backedge

loop.backedge:                                              ; preds = %loop.progress
  store i32 1, i32* %tmp7, align 4
  %tmp = add nuw i64 %tmp5, 1
  %tmp3 = icmp ult i64 %tmp, 1000
  br i1 %tmp3, label %loop, label %loop.progress1

loop:                                              ; preds = %loop.backedge, %entry
  %tmp5 = phi i64 [ %tmp, %loop.backedge ], [ 16, %entry ]
  %tmp6 = phi i64 [ %tmp5, %loop.backedge ], [ 15, %entry ]
  %tmp7 = getelementptr inbounds i32, i32* %p, i64 %tmp5
  %tmp8 = load i32, i32* %tmp7, align 4
  %tmp9 = add i32 %tmp8, -5
  store i32 %tmp9, i32* %tmp7, align 4
  br i1 false, label %never, label %loop.progress

never:                                             ; preds = %loop
  unreachable

loop.progress1:                                             ; preds = %loop.backedge
  ret void
}

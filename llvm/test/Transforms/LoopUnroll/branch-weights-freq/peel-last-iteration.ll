; RUN: opt -p "print<block-freq>,loop-unroll,print<block-freq>" -scev-cheap-expansion-budget=3 -S %s 2>&1 | FileCheck %s

define i32 @test_expansion_cost_2(i32 %start, i32 %end) !prof !0 {
entry:
  %sub = add i32 %end, -1
  br label %loop.header

loop.header:
  %iv = phi i32 [ %start, %entry ], [ %iv.next, %loop.latch ]
  %c = icmp eq i32 %iv, %sub
  br i1 %c, label %then, label %loop.latch, !prof !1

then:
  br label %loop.latch

loop.latch:
  %iv.next = add nsw i32 %iv, 1
  %ec = icmp eq i32 %iv.next, %end
  br i1 %ec, label %exit, label %loop.header, !prof !2

exit:
  ret i32 0
}

!0 = !{!"function_entry_count", i32 10}
!1 = !{!"branch_weights", i32 2, i32 3}
!2 = !{!"branch_weights", i32 1, i32 50}

; CHECK:        block-frequency-info: test_expansion_cost_2
; CHECK-NEXT:   entry: float = 1.0
; CHECK-NEXT:   loop.header: float = 51.0
; CHECK-NEXT:   then: float = 20.4
; CHECK-NEXT:   loop.latch: float = 51.0
; CHECK-NEXT:   exit: float = 1.0

; CHECK:       block-frequency-info: test_expansion_cost_2
; CHECK-NEXT:    entry: float = 1.0
; CHECK-NEXT:    entry.split: float = 0.98039
; CHECK-NEXT:    loop.header: float = 50.0
; CHECK-NEXT:    then: float = 20.0
; CHECK-NEXT:    loop.latch: float = 50.0
; CHECK-NEXT:    exit.peel.begin.loopexit: float = 0.98039
; CHECK-NEXT:    exit.peel.begin: float = 1.0
; CHECK-NEXT:    loop.header.peel: float = 1.0
; CHECK-NEXT:    then.peel: float = 0.4
; CHECK-NEXT:    loop.latch.peel: float = 1.0
; CHECK-NEXT:    exit.peel.next: float = 1.0
; CHECK-NEXT:    loop.header.peel.next: float = 1.0
; CHECK-NEXT:    exit: float = 1.0
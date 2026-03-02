; Disable this test in profcheck because the first run would cause profcheck to fail.
; REQUIRES: !profcheck
; RUN: opt -p "print<block-freq>,loop-unroll,print<block-freq>" -scev-cheap-expansion-budget=3 -S %s -profcheck-disable-metadata-fixes 2>&1 | FileCheck %s --check-prefixes=COMMON,BAD
; RUN: opt -p "print<block-freq>,loop-unroll,print<block-freq>" -scev-cheap-expansion-budget=3 -S %s 2>&1 | FileCheck %s --check-prefixes=COMMON,GOOD

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

; COMMON:        block-frequency-info: test_expansion_cost_2
; COMMON-NEXT:   entry: float = 1.0
; COMMON-NEXT:   loop.header: float = 51.0
; COMMON-NEXT:   then: float = 20.4
; COMMON-NEXT:   loop.latch: float = 51.0
; COMMON-NEXT:   exit: float = 1.0

; COMMON:       block-frequency-info: test_expansion_cost_2
; GOOD-NEXT:    entry: float = 1.0
; GOOD-NEXT:    entry.split: float = 0.98039
; GOOD-NEXT:    loop.header: float = 50.0
; GOOD-NEXT:    then: float = 20.0
; GOOD-NEXT:    loop.latch: float = 50.0
; GOOD-NEXT:    exit.peel.begin.loopexit: float = 0.98039
; GOOD-NEXT:    exit.peel.begin: float = 1.0
; GOOD-NEXT:    loop.header.peel: float = 1.0
; GOOD-NEXT:    then.peel: float = 0.4
; GOOD-NEXT:    loop.latch.peel: float = 1.0
; GOOD-NEXT:    exit.peel.next: float = 1.0
; GOOD-NEXT:    loop.header.peel.next: float = 1.0
; GOOD-NEXT:    exit: float = 1.0

; BAD-NEXT:  entry: float = 1.0
; BAD-NEXT:  entry.split: float = 0.625
; BAD-NEXT:  loop.header: float = 31.875
; BAD-NEXT:  then: float = 12.75
; BAD-NEXT:  loop.latch: float = 31.875
; BAD-NEXT:  exit.peel.begin.loopexit: float = 0.625
; BAD-NEXT:  exit.peel.begin: float = 1.0
; BAD-NEXT:  loop.header.peel: float = 1.0
; BAD-NEXT:  then.peel: float = 0.4
; BAD-NEXT:  loop.latch.peel: float = 1.0
; BAD-NEXT:  exit.peel.next: float = 1.0
; BAD-NEXT:  loop.header.peel.next: float = 1.0
; BAD-NEXT:  exit: float = 1.0
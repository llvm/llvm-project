; Check that the loop body frequency is maintained when LoopPeel both:
; - Peels one unconditional iteration after the loop.
; - Does not add a guard to sometimes skip the remaining loop because it has
;   proven the original loop always executes at least two iterations, which
;   become the initial iteration and the peeled iteration.

; DEFINE: %{exitWeight} =
; DEFINE: %{loopWeight} =
; DEFINE: %{loopFreqOld} =
; DEFINE: %{loopFreqNew} =

; DEFINE: %{run} = \
; DEFINE:   cp %s %t.ll && chmod +w %t.ll && \
; DEFINE:   echo '!0 = !{!"branch_weights", i32 %{exitWeight}, ' \
; DEFINE:       'i32 %{loopWeight}}' >> %t.ll && \
; DEFINE:   opt -p "print<block-freq>,loop-unroll,print<block-freq>" \
; DEFINE:       -unroll-full-max-count=0 -S %t.ll 2>&1 | \
; DEFINE:     FileCheck -DLOOP_FREQ_OLD='%{loopFreqOld}' \
; DEFINE:         -DLOOP_FREQ_NEW='%{loopFreqNew}' %s

; Branch weights give the original loop 10 iterations.  We expect that
; loopFreqOld = loopFreqNew + 1.
; REDEFINE: %{exitWeight} = 1
; REDEFINE: %{loopWeight} = 9
; REDEFINE: %{loopFreqOld} = 10.0
; REDEFINE: %{loopFreqNew} = 9.0
; RUN: %{run}

; Branch weights give the original loop 2 iterations.  We expect that
; loopFreqOld = loopFreqNew + 1.
; REDEFINE: %{exitWeight} = 1
; REDEFINE: %{loopWeight} = 1
; REDEFINE: %{loopFreqOld} = 2.0
; REDEFINE: %{loopFreqNew} = 1.0
; RUN: %{run}

; Branch weights give the original loop 1 iteration, but LoopPeel proved it has
; at least 2.  There is no loop probability that produces a frequency below 1,
; so the original total frequency cannot be maintained.
; REDEFINE: %{exitWeight} = 1
; REDEFINE: %{loopWeight} = 0
; REDEFINE: %{loopFreqOld} = 1.0
; REDEFINE: %{loopFreqNew} = 1.0
; RUN: %{run}

; Branch weights say the original loop is infinite, maximizing the frequency,
; so LoopPeel does not try to decrement it.
; REDEFINE: %{exitWeight} = 0
; REDEFINE: %{loopWeight} = 1
; REDEFINE: %{loopFreqOld} = 2147483647.8
; REDEFINE: %{loopFreqNew} = 2147483647.8
; RUN: %{run}

; Everything other than loop should be 1.0 because it is reached once.
;
;      CHECK: block-frequency-info: test
; CHECK-NEXT: - entry: float = 1.0,
; CHECK-NEXT: - loop: float = [[LOOP_FREQ_OLD]],
; CHECK-NEXT: - exit: float = 1.0,
;
;      CHECK: block-frequency-info: test
; CHECK-NEXT: - entry: float = 1.0,
; CHECK-NEXT: - loop: float = [[LOOP_FREQ_NEW]],
; CHECK-NEXT: - exit.peel.begin: float = 1.0,
; CHECK-NEXT: - loop.peel: float = 1.0,
; CHECK-NEXT: - exit.peel.next: float = 1.0,
; CHECK-NEXT: - loop.peel.next: float = 1.0,
; CHECK-NEXT: - exit: float = 1.0,

declare void @f(i32)

define void @test() {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %inc, %loop ]
  %isLast = icmp eq i32 %i, 20
  %sel = select i1 %isLast, i32 1, i32 0
  call void @f(i32 %sel)
  %inc = add i32 %i, 1
  %isLast1 = icmp eq i32 %i, 20
  br i1 %isLast1, label %exit, label %loop, !prof !0

exit:
  ret void
}

; Test branch weight metadata, estimated trip count metadata, and block
; frequencies after partial loop unrolling without -unroll-runtime such that
; some iterations' latches become unconditional, which often contradicts the
; original branch weights.
;
; (unroll-complete.ll tests complete loop unrolling, in which the final unrolled
; iteration unconditionally exits (backedge removed).  Here, we include cases
; where the final iteration's latch unconditionally continues instead.)
;
; For each case, we check:
; - Iteration frequencies
;   - When each is multiplied by the number of original loop bodies that execute
;     within it, they should sum to almost exactly the original loop body
;     frequency.
;   - The only exception is an impossibly high or low original frequency (e.g.,
;     due to bad profile data), for which there exist no new branch weights that
;     can yield that frequency sum.  In those cases, we expect the maximum or
;     minimum possible frequency.
; - CFGs
;   - We verify which branch weights go with which branches and that we did not
;     overlook any other branch weights (no extra !prof or branch_weights).
;   - We also check the number of original loop bodies (represented by a call to
;     @f) that appear within each unrolled iteration.
; - Branch weight metadata
;   - Checking frequencies already checks whether the branch weights have the
;     expected effect, but we also want to check that we get uniform
;     probabilities/weights (same !prof) across the unrolled iteration latches
;     when expected.
; - llvm.loop.estimated_trip_count
;   - It must be the number of iterations of the unrolled loop required for the
;     original loop body to reach its original frequency.
;   - It must not be blindly computed from any new latch branch weights.

; ------------------------------------------------------------------------------
; Define LIT substitutions.
;
; For verifying that the test code produces the original loop body frequency we
; expect.
; DEFINE: %{bf-fc} = opt %t.ll -S -passes='print<block-freq>' 2>&1 | \
; DEFINE:   FileCheck %s -check-prefixes
;
; For checking the unrolled loop:
; DEFINE: %{ur-bf} = opt %t.ll -S -passes='loop-unroll,print<block-freq>' 2>&1
; DEFINE: %{fc} = FileCheck %s \
; DEFINE:     -implicit-check-not='llvm.loop.estimated_trip_count' \
; DEFINE:     -implicit-check-not='!prof' \
; DEFINE:     -implicit-check-not='branch_weights' \
; DEFINE:     -implicit-check-not='call void @f' -check-prefixes

; ------------------------------------------------------------------------------
; Check cases when the original loop's number of iterations is a run-time
; determined multiple of 10 and the original loop body frequency is 10.
;
;   RUN: sed -e s/@N@/%mul10/ -e s/@W@/9/ %s > %t.ll
;
; At compile time, possibilities for that value always include unroll count x 10
; x N for any integer N >= 1, so the unrolled loop's backedge always remains
; conditional, so we check cases where it becomes unconditional later in this
; test file with the CONST4 config.
;
; Check the original loop body frequency.
;
;   RUN: %{bf-fc} MULT-ORIG
;   MULT-ORIG: - do.body: float = 10.0,
;
; When the unroll count is odd, every iteration's latch remains conditional, so
; their original probabilities are not contradicted.  That is, the original loop
; latch's branch weights remain on all unrolled iterations' latches.
;
;   RUN: %{ur-bf} -unroll-count=3 -unroll-uniform-weights | %{fc} MULT3
;   RUN: %{ur-bf} -unroll-count=3 | %{fc} MULT3
;
;   Sums to approximately the original loop body frequency, 10.
;   MULT3: - do.body: float = 3.69,
;   MULT3: - do.body.1: float = 3.321,
;   MULT3: - do.body.2: float = 2.9889,
;
;   MULT3: call void @f
;   MULT3: br i1 %{{.*}}, label %do.body.1, label %do.end, !prof !0
;   MULT3: call void @f
;   MULT3: br i1 %{{.*}}, label %do.body.2, label %do.end, !prof !0
;   MULT3: call void @f
;   MULT3: br i1 %{{.*}}, label %do.body, label %do.end, !prof !0, !llvm.loop !1
;
;   MULT3: !0 = !{!"branch_weights", i32 9, i32 1}
;   MULT3: !1 = distinct !{!1, !2, !3}
;   MULT3: !2 = !{!"llvm.loop.estimated_trip_count", i32 4}
;   MULT3: !3 = !{!"llvm.loop.unroll.disable"}
;
; When the unroll count is even, odd-numbered unrolled iterations become
; unconditional, so branch weights must be adjusted.
;
;   -unroll-count=2, so there is 1 remaining conditional latch, so the
;   implementation can compute uniform weights by solving a linear equation.
;   Thus, -unroll-uniform-weights has no effect.
;
;     RUN: %{ur-bf} -unroll-count=2 -unroll-uniform-weights | %{fc} MULT2
;     RUN: %{ur-bf} -unroll-count=2 | %{fc} MULT2
;
;     Multiply by 2 to get the original loop body frequency, 10.
;     MULT2: - do.body: float = 5.0,
;
;     MULT2:     call void @f
;     MULT2-NOT: br
;     MULT2:     call void @f
;     MULT2:     br i1 %{{.*}}, label %do.body, label %do.end, !prof !0, !llvm.loop !1{{$}}
;
;     The branch weights imply the estimated trip count is
;     (1717986918+429496730)/429496730 = approximately (8+2)/2 = 5.
;     MULT2: !0 = !{!"branch_weights", i32 1717986918, i32 429496730}
;     MULT2: !1 = distinct !{!1, !2, !3}
;     MULT2: !2 = !{!"llvm.loop.estimated_trip_count", i32 5}
;     MULT2: !3 = !{!"llvm.loop.unroll.disable"}
;
;   -unroll-count=4, so there are 2 remaining conditional latches, so the
;   implementation can compute uniform weights using the quadratic formula.
;   Thus, -unroll-uniform-weights has no effect.
;
;     RUN: %{ur-bf} -unroll-count=4 -unroll-uniform-weights | %{fc} MULT4
;     RUN: %{ur-bf} -unroll-count=4 | %{fc} MULT4
;
;     Multiply by 2 and sum to get the original loop body frequency, 10.
;     MULT4: - do.body: float = 2.7778,
;     MULT4: - do.body.2: float = 2.2222,
;
;     MULT4:     call void @f
;     MULT4-NOT: br
;     MULT4:     call void @f
;     MULT4:     br i1 %{{.*}}, label %do.body.2, label %do.end, !prof !0
;     MULT4:     call void @f
;     MULT4-NOT: br
;     MULT4:     call void @f
;     MULT4:     br i1 %{{.*}}, label %do.body, label %do.end, !prof !0, !llvm.loop !1
;
;     MULT4 is like applying -unroll-count=2 to MULT2 without converting any
;     more conditional latches to unconditional, so MULT2's branch weights work.
;     MULT4: !0 = !{!"branch_weights", i32 1717986918, i32 429496730}
;     MULT4: !1 = distinct !{!1, !2, !3}
;     MULT4: !2 = !{!"llvm.loop.estimated_trip_count", i32 3}
;     MULT4: !3 = !{!"llvm.loop.unroll.disable"}
;
;   -unroll-count=6, so there are 3 remaining conditional latches, the lowest
;   number where the implementation cannot compute uniform weights using a
;   simple formula.  Thus, this is our first case where -unroll-uniform-weights
;   matters.
;
;     RUN: %{ur-bf} -unroll-count=6 -unroll-uniform-weights | %{fc} MULT6,MUNIF6
;     RUN: %{ur-bf} -unroll-count=6 | %{fc} MULT6,MFAST6
;
;     For either MUNIF or MFAST, multiply by 2 and sum to get the original loop
;     body frequency, 10.
;     MUNIF6: - do.body: float = 2.0492,
;     MUNIF6: - do.body.2: float = 1.6393,
;     MUNIF6: - do.body.4: float = 1.3115,
;     MFAST6: - do.body: float = 2.1956,
;     MFAST6: - do.body.2: float = 1.476,
;     MFAST6: - do.body.4: float = 1.3284,
;
;     MULT6:       call void @f
;     MULT6-NOT:   br
;     MULT6:       call void @f
;     MULT6:       br i1 %{{.*}}, label %do.body.2, label %do.end, !prof !0
;     MULT6:       call void @f
;     MULT6-NOT:   br
;     MULT6:       call void @f
;     MULT6:       br i1 %{{.*}}, label %do.body.4, label %do.end,
;     MUNIF6-SAME:   !prof !0
;     MFAST6-SAME:   !prof !1
;     MULT6:       call void @f
;     MULT6-NOT:   br
;     MULT6:       call void @f
;     MULT6:       br i1 %{{.*}}, label %do.body, label %do.end,
;     MUNIF6-SAME:   !prof !0, !llvm.loop !1
;     MFAST6-SAME:   !prof !1, !llvm.loop !2
;
;     MUNIF6 is like applying -unroll-count=3 to MULT2 without converting any
;     additional conditional latches to unconditional, so (approximately)
;     MULT2's branch weights make sense.
;     MUNIF6: !0 = !{!"branch_weights", i32 1717986944, i32 429496704}
;     MUNIF6: !1 = distinct !{!1, !2, !3}
;     MUNIF6: !2 = !{!"llvm.loop.estimated_trip_count", i32 2}
;     MUNIF6: !3 = !{!"llvm.loop.unroll.disable"}
;
;     There are 3 conditional latches remaining, so MFAST6 adjusts the first and
;     leaves the second two with the original loop's branch weights.
;     MFAST6: !0 = !{!"branch_weights", i32 1443686486, i32 703797162}
;     MFAST6: !1 = !{!"branch_weights", i32 9, i32 1}
;     MFAST6: !2 = distinct !{!2, !3, !4}
;     MFAST6: !3 = !{!"llvm.loop.estimated_trip_count", i32 2}
;     MFAST6: !4 = !{!"llvm.loop.unroll.disable"}

; ------------------------------------------------------------------------------
; Check case when the original loop's number of iterations is a run-time
; determined multiple of 10, the unroll count is even so that odd-numbered
; unrolled iterations become unconditional, and the original loop body frequency
; is 1, which is impossibly low.  This case is important to ensure the
; implementation does not malfunction by trying to use negative and possibly
; infinite probabilities to reach the original loop body frequency.
;
;   RUN: sed -e s/@N@/%mul10/ -e s/@W@/0/ %s > %t.ll
;
; Check the original loop body frequency.
;
;   RUN: %{bf-fc} LOW-ORIG
;   LOW-ORIG: - do.body: float = 1.0,
;
; -unroll-count=2, so there is 1 remaining conditional latch.  The
; implementation tries to compute uniform weights by solving a linear equation
; but ultimately sets the latch's probability to zero.
;
;   RUN: %{ur-bf} -unroll-count=2 -unroll-uniform-weights | %{fc} LOW2
;   RUN: %{ur-bf} -unroll-count=2 | %{fc} LOW2
;
;   Multiply by 2, but the result is greater than the original loop body
;   frequency, 1, which is impossibly low.
;   LOW2: - do.body: float = 1.0,
;
;   LOW2:     call void @f
;   LOW2-NOT: br
;   LOW2:     call void @f
;   LOW2:     br i1 %{{.*}}, label %do.body, label %do.end, !prof !0, !llvm.loop !1{{$}}
;
;   LOW2: !0 = !{!"branch_weights", i32 0, i32 -2147483648}
;   LOW2: !1 = distinct !{!1, !2, !3}
;   LOW2: !2 = !{!"llvm.loop.estimated_trip_count", i32 1}
;   LOW2: !3 = !{!"llvm.loop.unroll.disable"}
;
; -unroll-count=4, so there are 2 remaining conditional latches.  The
; implementation tries to compute uniform weights using the quadratic formula
; but ultimately sets both latches' probabilities to zero.
;
;   RUN: %{ur-bf} -unroll-count=4 -unroll-uniform-weights | %{fc} LOW4
;   RUN: %{ur-bf} -unroll-count=4 | %{fc} LOW4
;
;   Multiply by 2 and sum, but the result is greater than the original loop body
;   frequency, 1, which is impossibly low.
;   LOW4: - do.body: float = 1.0,
;   LOW4: - do.body.2: float = 0.0{{(0000[0-9]*)?}},
;
;   LOW4:     call void @f
;   LOW4-NOT: br
;   LOW4:     call void @f
;   LOW4:     br i1 %{{.*}}, label %do.body.2, label %do.end, !prof !0
;   LOW4:     call void @f
;   LOW4-NOT: br
;   LOW4:     call void @f
;   LOW4:     br i1 %{{.*}}, label %do.body, label %do.end, !prof !0, !llvm.loop !1
;
;   LOW4: !0 = !{!"branch_weights", i32 0, i32 -2147483648}
;   LOW4: !1 = distinct !{!1, !2, !3}
;   LOW4: !2 = !{!"llvm.loop.estimated_trip_count", i32 1}
;   LOW4: !3 = !{!"llvm.loop.unroll.disable"}
;
; -unroll-count=6, so there are 3 remaining conditional latches.  The
; implementation cannot compute uniform weights using a simple formula, and
; ultimately it must set all those latches' probabilities to zero.  If not
; -unroll-uniform-weights, then the implementation will face a new stumbling
; block starting at the second latch: reaching the remaining iterations already
; has a zero probability due to the zero probability set at the first latch, so
; the required probability could accidentally be computed as negative infinity.
;
;   RUN: %{ur-bf} -unroll-count=6 -unroll-uniform-weights | %{fc} LOW6
;   RUN: %{ur-bf} -unroll-count=6 | %{fc} LOW6
;
;   Multiply by 2 and sum, but the result is greater than the original loop body
;   frequency, 1, which is impossibly low.
;   LOW6: - do.body: float = 1.0,
;   LOW6: - do.body.2: float = 0.0{{(0000[0-9]*)?}},
;   LOW6: - do.body.4: float = 0.0{{(0000[0-9]*)?}},
;
;   LOW6:     call void @f
;   LOW6-NOT: br
;   LOW6:     call void @f
;   LOW6:     br i1 %{{.*}}, label %do.body.2, label %do.end, !prof !0
;   LOW6:     call void @f
;   LOW6-NOT: br
;   LOW6:     call void @f
;   LOW6:     br i1 %{{.*}}, label %do.body.4, label %do.end, !prof !0
;   LOW6:     call void @f
;   LOW6-NOT: br
;   LOW6:     call void @f
;   LOW6:     br i1 %{{.*}}, label %do.body, label %do.end, !prof !0, !llvm.loop !1
;
;   LOW6: !0 = !{!"branch_weights", i32 0, i32 -2147483648}
;   LOW6: !1 = distinct !{!1, !2, !3}
;   LOW6: !2 = !{!"llvm.loop.estimated_trip_count", i32 1}
;   LOW6: !3 = !{!"llvm.loop.unroll.disable"}

; ------------------------------------------------------------------------------
; Check cases when the original loop's number of iterations is a constant 10 and
; the original loop body frequency is 10.
;
;   RUN: sed -e s/@N@/10/g -e s/@W@/9/ %s > %t.ll
;
; Because we test only partial unrolling, there is always exactly one unrolled
; iteration that can possibly exit, so only its latch can remain conditional.
; Because there is only one, its branch weights can be computed with a simple
; formula, and -unroll-uniform-weights does not matter.
;
; Check the original loop body frequency.
;
;   RUN: %{bf-fc} CONST-ORIG
;   CONST-ORIG: - do.body: float = 10.0,
;
; Check when the unrolled loop's backedge remains conditional.
;
;   RUN: %{ur-bf} -unroll-count=2 -unroll-uniform-weights | %{fc} CONST2
;   RUN: %{ur-bf} -unroll-count=2 | %{fc} CONST2
;
;   Multiply by 2 to get the original loop body frequency, 10.
;   CONST2: - do.body: float = 5.0,
;
;   CONST2:     call void @f
;   CONST2-NOT: br:
;   CONST2:     call void @f
;   CONST2:     br i1 %{{.*}}, label %do.body, label %do.end, !prof !0, !llvm.loop !1
;
;   Like MULT2.
;   CONST2: !0 = !{!"branch_weights", i32 1717986918, i32 429496730}
;   CONST2: !1 = distinct !{!1, !2, !3}
;   CONST2: !2 = !{!"llvm.loop.estimated_trip_count", i32 5}
;   CONST2: !3 = !{!"llvm.loop.unroll.disable"}
;
; Check when the unrolled loop's backedge unconditionally continues.
;
;   RUN: %{ur-bf} -unroll-count=4 -unroll-uniform-weights | %{fc} CONST4
;   RUN: %{ur-bf} -unroll-count=4 | %{fc} CONST4
;
;   Multiply by 2 and sum to get the original loop body frequency, 10.
;   CONST4: - do.body: float = 3.0,
;   CONST4: - do.body.2: float = 2.0,
;
;   CONST4:     call void @f
;   CONST4-NOT: br
;   CONST4:     call void @f
;   CONST4:     br i1 %{{.*}}, label %do.body.2, label %do.end, !prof !0
;   CONST4:     call void @f
;   CONST4-NOT: br
;   CONST4:     call void @f
;   CONST4:     br label %do.body, !llvm.loop !1
;
;   There is no llvm.loop.estimated_trip_count because the unrolled loop's latch
;   in do.body.2 unconditionally continues.  The branch weights on do.body's
;   branch imply do.body continues twice and then exits once, thus executing the
;   original loop body 10 times.
;   CONST4: !0 = !{!"branch_weights", i32 1431655765, i32 715827883}
;   CONST4: !1 = distinct !{!1, !2}
;   CONST4: !2 = !{!"llvm.loop.unroll.disable"}

declare void @f(i32)

define void @test(i32 %n) {
entry:
  %mul10 = mul i32 %n, 10
  br label %do.body

do.body:
  %i = phi i32 [ 0, %entry ], [ %next, %do.body ]
  call void @f(i32 %i)
  %next = add i32 %i, 1
  %c = icmp ne i32 %next, @N@
  br i1 %c, label %do.body, label %do.end, !prof !0

do.end:
  ret void
}

; Loop body frequency is @W@ + 1.
!0 = !{!"branch_weights", i32 @W@, i32 1}

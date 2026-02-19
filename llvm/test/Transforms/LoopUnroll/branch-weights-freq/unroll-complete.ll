; Test branch weight metadata, estimated trip count metadata, and block
; frequencies after complete loop unrolling.  The final unrolled iteration
; unconditionally exits (backedge removed), and other unrolled iterations'
; latches might unconditionally continue.  Either contradicts the original
; branch weights.
;
; (unroll-partial-unconditional-latch.ll tests partial unrolling cases,
; including cases where the latch of any iteration, including the final, might
; unconditionally continue.)
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
; - llvm.loop.estimated_trip_count:
;   - There should be none because loops are completely unrolled.

; ------------------------------------------------------------------------------
; Define LIT substitutions.
;
; Before using the following lit substitutions, sed should be called to replace
; these parameters in %s to produce %t.ll:
; - @I_0@ is the starting value for the original loop's induction variable.
; - @MIN@ and @MAX@ are the compile-time known minimum and maximum for the
;   number of original loop iterations, regardless of @I_0@.
; - @W@ is the branch weight for the original loop's backedge.  That value plus
;   1 is the original loop body frequency because the exit branch weight is 1.
;
; For verifying that the test code produces the original loop body frequency we
; expect.
; DEFINE: %{bf-fc} = opt %t.ll -S -passes='print<block-freq>' 2>&1 | \
; DEFINE:   FileCheck %s -check-prefixes
;
; For checking the unrolled loop.
; DEFINE: %{ur-bf} = opt %t.ll -S -passes='loop-unroll,print<block-freq>' 2>&1
; DEFINE: %{fc} = FileCheck %s \
; DEFINE:     -implicit-check-not='llvm.loop.estimated_trip_count' \
; DEFINE:     -implicit-check-not='!prof' \
; DEFINE:     -implicit-check-not='branch_weights' \
; DEFINE:     -implicit-check-not='call void @f' -check-prefixes

; ------------------------------------------------------------------------------
; Check 1 max iteration:
; - Unroll count of >=1 should always produce complete unrolling.
; - That produces 0 unrolled iteration latches, so there are no branch weights
;   to compute.  Thus, -unroll-uniform-weights has no effect.
;
; Original loop body frequency is 2 (loop weight 1), which is impossibly high.
;
;   RUN: sed -e s/@MAX@/1/ -e s/@W@/1/ -e s/@MIN@/1/ -e s/@I_0@/0/ %s > %t.ll
;   RUN: %{bf-fc} ORIG1210
;   RUN: %{ur-bf} -unroll-count=1 -unroll-uniform-weights | %{fc} UR1210
;   RUN: %{ur-bf} -unroll-count=2 -unroll-uniform-weights | %{fc} UR1210
;   RUN: %{ur-bf} -unroll-count=1 | %{fc} UR1210
;   RUN: %{ur-bf} -unroll-count=2 | %{fc} UR1210
;
;   The new do.body is less than the old do.body, which is impossibly high.
;   ORIG1210: - do.body: float = 2.0,
;   UR1210:   - do.body: float = 1.0,
;
;   UR1210: call void @f
;
; Original loop body frequency is 1 (loop weight 0).
;
;   RUN: sed -e s/@MAX@/1/ -e s/@W@/0/ -e s/@MIN@/1/ -e s/@I_0@/0/ %s > %t.ll
;   RUN: %{bf-fc} ORIG1110
;   RUN: %{ur-bf} -unroll-count=1 -unroll-uniform-weights | %{fc} UR1110
;   RUN: %{ur-bf} -unroll-count=2 -unroll-uniform-weights | %{fc} UR1110
;   RUN: %{ur-bf} -unroll-count=1 | %{fc} UR1110
;   RUN: %{ur-bf} -unroll-count=2 | %{fc} UR1110
;
;   The the new do.body equals the old do.body.
;   ORIG1110: - do.body: float = 1.0,
;   UR1110:   - do.body: float = 1.0,
;
;   UR1110: call void @f

; ------------------------------------------------------------------------------
; Check 2 max iterations:
; - Unroll count of >=2 should always produce complete unrolling.
; - That produces <=1 unrolled iteration latch, so the implementation can
;   compute uniform weights by solving, at worst, a linear equation.  Thus,
;   -unroll-uniform-weights has no effect.
;
; Original loop body frequency is 3 (loop weight 2), which is impossibly high.
;
;   First use a variable iteration count so that the sole non-final unrolled
;   iteration's latch remains conditional.
;
;     RUN: sed -e s/@MAX@/2/ -e s/@W@/2/ -e s/@MIN@/1/ -e s/@I_0@/0/ %s > %t.ll
;     RUN: %{bf-fc} ORIG2310
;     RUN: %{ur-bf} -unroll-count=2 -unroll-uniform-weights | %{fc} UR2310
;     RUN: %{ur-bf} -unroll-count=3 -unroll-uniform-weights | %{fc} UR2310
;     RUN: %{ur-bf} -unroll-count=2 | %{fc} UR2310
;     RUN: %{ur-bf} -unroll-count=3 | %{fc} UR2310
;
;     The sum of the new do.body* cannot reach the old do.body, which is
;     impossibly high.
;     ORIG2310: - do.body: float = 3.0,
;     UR2310:   - do.body: float = 1.0,
;     UR2310:   - do.body.1: float = 1.0,
;
;     The sole probability is maximized to try to reach the original frequency.
;     UR2310: call void @f
;     UR2310: br i1 %{{.*}}, label %do.end, label %do.body.1, !prof !0
;     UR2310: call void @f
;     UR2310: br label %do.end
;     UR2310: !0 = !{!"branch_weights", i32 0, i32 -2147483648}
;
;   Now use a constant iteration count so that the sole non-final unrolled
;   iteration's latch unconditionally continues.
;
;     RUN: sed -e s/@MAX@/2/ -e s/@W@/2/ -e s/@MIN@/2/ -e s/@I_0@/0/ %s > %t.ll
;     RUN: %{bf-fc} ORIG2320
;     RUN: %{ur-bf} -unroll-count=2 -unroll-uniform-weights | %{fc} UR2320
;     RUN: %{ur-bf} -unroll-count=3 -unroll-uniform-weights | %{fc} UR2320
;     RUN: %{ur-bf} -unroll-count=2 | %{fc} UR2320
;     RUN: %{ur-bf} -unroll-count=3 | %{fc} UR2320
;
;     The new do.body contains 2 of the original loop's iterations, so multiply
;     it by 2, which is less than the old do.body, which is impossibly high.
;     ORIG2320: - do.body: float = 3.0,
;     UR2320:   - do.body: float = 1.0,
;
;     UR2320:     call void @f
;     UR2320-NOT: br
;     UR2320:     call void @f
;     UR2320:     ret void
;
; Original loop body frequency is 2 (loop weight 1).
;
;   First use a variable iteration count so that the sole non-final unrolled
;   iteration's latch remains conditional.
;
;     RUN: sed -e s/@MAX@/2/ -e s/@W@/1/ -e s/@MIN@/1/ -e s/@I_0@/0/ %s > %t.ll
;     RUN: %{bf-fc} ORIG2210
;     RUN: %{ur-bf} -unroll-count=2 -unroll-uniform-weights | %{fc} UR2210
;     RUN: %{ur-bf} -unroll-count=3 -unroll-uniform-weights | %{fc} UR2210
;     RUN: %{ur-bf} -unroll-count=2 | %{fc} UR2210
;     RUN: %{ur-bf} -unroll-count=3 | %{fc} UR2210
;
;     The sum of the new do.body* is the old do.body.
;     ORIG2210: - do.body: float = 2.0,
;     UR2210:   - do.body: float = 1.0,
;     UR2210:   - do.body.1: float = 1.0,
;
;     UR2210: call void @f
;     UR2210: br i1 %{{.*}}, label %do.end, label %do.body.1, !prof !0
;     UR2210: call void @f
;     UR2210: br label %do.end
;     UR2210: !0 = !{!"branch_weights", i32 0, i32 -2147483648}
;
;   Now use a constant iteration count so that the sole non-final unrolled
;   iteration's latch unconditionally continues.
;
;     RUN: sed -e s/@MAX@/2/ -e s/@W@/1/ -e s/@MIN@/2/ -e s/@I_0@/0/ %s > %t.ll
;     RUN: %{bf-fc} ORIG2220
;     RUN: %{ur-bf} -unroll-count=2 -unroll-uniform-weights | %{fc} UR2220
;     RUN: %{ur-bf} -unroll-count=3 -unroll-uniform-weights | %{fc} UR2220
;     RUN: %{ur-bf} -unroll-count=2 | %{fc} UR2220
;     RUN: %{ur-bf} -unroll-count=3 | %{fc} UR2220
;
;     The new do.body contains 2 of the original loop's iterations, so multiply
;     it by 2 to get the old do.body.
;     ORIG2220: - do.body: float = 2.0,
;     UR2220:   - do.body: float = 1.0,
;
;     UR2220:     call void @f
;     UR2220-NOT: br
;     UR2220:     call void @f
;     UR2220:     ret void
;
; Original loop body frequency is 1 (loop weight 0).
;
;   First use a variable iteration count so that the sole non-final unrolled
;   iteration's latch remains conditional.
;
;     RUN: sed -e s/@MAX@/2/ -e s/@W@/0/ -e s/@MIN@/1/ -e s/@I_0@/0/ %s > %t.ll
;     RUN: %{bf-fc} ORIG2110
;     RUN: %{ur-bf} -unroll-count=2 -unroll-uniform-weights | %{fc} UR2110
;     RUN: %{ur-bf} -unroll-count=3 -unroll-uniform-weights | %{fc} UR2110
;     RUN: %{ur-bf} -unroll-count=2 | %{fc} UR2110
;     RUN: %{ur-bf} -unroll-count=3 | %{fc} UR2110
;
;     The sum of the new do.body* is approximately the old do.body.
;     ORIG2110: - do.body: float = 1.0,
;     UR2110:   - do.body: float = 1.0,
;     UR2110:   - do.body.1: float = 0.0{{(0000[0-9]*)?}},
;
;     UR2110: call void @f
;     UR2110: br i1 %{{.*}}, label %do.end, label %do.body.1, !prof !0
;     UR2110: call void @f
;     UR2110: br label %do.end
;     UR2110: !0 = !{!"branch_weights", i32 1, i32 0}
;
;   Now use a constant iteration count so that the sole non-final unrolled
;   iteration's latch unconditionally continues.
;
;     RUN: sed -e s/@MAX@/2/ -e s/@W@/0/ -e s/@MIN@/2/ -e s/@I_0@/0/ %s > %t.ll
;     RUN: %{bf-fc} ORIG2120
;     RUN: %{ur-bf} -unroll-count=2 -unroll-uniform-weights | %{fc} UR2120
;     RUN: %{ur-bf} -unroll-count=3 -unroll-uniform-weights | %{fc} UR2120
;     RUN: %{ur-bf} -unroll-count=2 | %{fc} UR2120
;     RUN: %{ur-bf} -unroll-count=3 | %{fc} UR2120
;
;     The new do.body contains 2 of the original loop's iterations, so multiply
;     it by 2, which is greater than the old do.body, which is impossibly low.
;     ORIG2120: - do.body: float = 1.0,
;     UR2120:   - do.body: float = 1.0,
;
;     UR2120:     call void @f
;     UR2220-NOT: br
;     UR2120:     call void @f
;     UR2120:     ret void

; ------------------------------------------------------------------------------
; Check 3 max iterations:
; - Unroll count of >=3 should always produce complete unrolling.
; - That produces <=2 unrolled iteration latches, so the implementation can
;   compute uniform weights solving, at worst, a quadratic equation.  Thus,
;   -unroll-uniform-weights has no effect.
;
; Original loop body frequency is 4 (loop weight 3), which is impossibly high.
;
;   First use a variable iteration count so that all non-final unrolled
;   iterations' latches remain conditional.
;
;     RUN: sed -e s/@MAX@/3/ -e s/@W@/3/ -e s/@MIN@/1/ -e s/@I_0@/0/ %s > %t.ll
;     RUN: %{bf-fc} ORIG3410
;     RUN: %{ur-bf} -unroll-count=3 -unroll-uniform-weights | %{fc} UR3410
;     RUN: %{ur-bf} -unroll-count=4 -unroll-uniform-weights | %{fc} UR3410
;     RUN: %{ur-bf} -unroll-count=3 | %{fc} UR3410
;     RUN: %{ur-bf} -unroll-count=4 | %{fc} UR3410
;
;     The sum of the new do.body* cannot reach the old do.body, which is
;     impossibly high.
;     ORIG3410: - do.body: float = 4.0,
;     UR3410:   - do.body: float = 1.0,
;     UR3410:   - do.body.1: float = 1.0,
;     UR3410:   - do.body.2: float = 1.0,
;
;     The probabilities are maximized to try to reach the original frequency.
;     UR3410: call void @f
;     UR3410: br i1 %{{.*}}, label %do.end, label %do.body.1, !prof !0
;     UR3410: call void @f
;     UR3410: br i1 %{{.*}}, label %do.end, label %do.body.2, !prof !0
;     UR3410: call void @f
;     UR3410: br label %do.end
;     UR3410: !0 = !{!"branch_weights", i32 0, i32 -2147483648}
;
;   Now use a constant iteration count so that all non-final unrolled
;   iterations' latches unconditionally continue.
;
;     RUN: sed -e s/@MAX@/3/ -e s/@W@/3/ -e s/@MIN@/3/ -e s/@I_0@/0/ %s > %t.ll
;     RUN: %{bf-fc} ORIG3430
;     RUN: %{ur-bf} -unroll-count=3 -unroll-uniform-weights | %{fc} UR3430
;     RUN: %{ur-bf} -unroll-count=4 -unroll-uniform-weights | %{fc} UR3430
;     RUN: %{ur-bf} -unroll-count=3 | %{fc} UR3430
;     RUN: %{ur-bf} -unroll-count=4 | %{fc} UR3430
;
;     The new do.body contains 3 of the original loop's iterations, so multiply
;     it by 3, which is less than the old do.body, which is impossibly high.
;     ORIG3430: - do.body: float = 4.0,
;     UR3430:   - do.body: float = 1.0,
;
;     UR3430:     call void @f
;     UR3430-NOT: br
;     UR3430:     call void @f
;     UR3430-NOT: br
;     UR3430:     call void @f
;     UR3430:     ret void
;
;   Use a constant iteration count but now the loop upper bound computation can
;   overflow.  When it does, the loop induction variable is greater than it
;   immediately, so the initial unrolled iteration's latch remains conditional.
;
;     RUN: sed -e s/@MAX@/3/ -e s/@W@/3/ -e s/@MIN@/3/ -e s/@I_0@/%x/ %s > %t.ll
;     RUN: %{bf-fc} ORIG343x
;     RUN: %{ur-bf} -unroll-count=3 -unroll-uniform-weights | %{fc} UR343x
;     RUN: %{ur-bf} -unroll-count=4 -unroll-uniform-weights | %{fc} UR343x
;     RUN: %{ur-bf} -unroll-count=3 | %{fc} UR343x
;     RUN: %{ur-bf} -unroll-count=4 | %{fc} UR343x
;
;     The new do.body.1 contains 2 of the original loop's iterations, so
;     multiply it by 2, and add the new do.body, but that sum is less than the
;     old do.body, which is impossibly high.
;     ORIG343x: - do.body: float = 4.0,
;     UR343x:   - do.body: float = 1.0,
;     UR343x:   - do.body.1: float = 1.0,
;
;     The sole probability is maximized to try to reach the original frequency.
;     UR343x:     call void @f
;     UR343x:     br i1 %{{.*}}, label %do.end, label %do.body.1, !prof !0
;     UR343x:     call void @f
;     UR343x-NOT: br
;     UR343x:     call void @f
;     UR343x:     ret void
;     UR343x:     !0 = !{!"branch_weights", i32 0, i32 -2147483648}
;
; Original loop body frequency is 3 (loop weight 2).
;
;   First use a variable iteration count so that all non-final unrolled
;   iterations' latches remain conditional.
;
;     RUN: sed -e s/@MAX@/3/ -e s/@W@/2/ -e s/@MIN@/1/ -e s/@I_0@/0/ %s > %t.ll
;     RUN: %{bf-fc} ORIG3310
;     RUN: %{ur-bf} -unroll-count=3 -unroll-uniform-weights | %{fc} UR3310
;     RUN: %{ur-bf} -unroll-count=4 -unroll-uniform-weights | %{fc} UR3310
;     RUN: %{ur-bf} -unroll-count=3 | %{fc} UR3310
;     RUN: %{ur-bf} -unroll-count=4 | %{fc} UR3310
;
;     The sum of the new do.body* is the old do.body.
;     ORIG3310: - do.body: float = 3.0,
;     UR3310:   - do.body: float = 1.0,
;     UR3310:   - do.body.1: float = 1.0,
;     UR3310:   - do.body.2: float = 1.0,
;
;     UR3310: call void @f
;     UR3310: br i1 %{{.*}}, label %do.end, label %do.body.1, !prof !0
;     UR3310: call void @f
;     UR3310: br i1 %{{.*}}, label %do.end, label %do.body.2, !prof !0
;     UR3310: call void @f
;     UR3310: br label %do.end
;     UR3310: !0 = !{!"branch_weights", i32 1, i32 2147483647}
;
;   Now use a constant iteration count so that all non-final unrolled
;   iterations' latches unconditionally continue.
;
;     RUN: sed -e s/@MAX@/3/ -e s/@W@/2/ -e s/@MIN@/3/ -e s/@I_0@/0/ %s > %t.ll
;     RUN: %{bf-fc} ORIG3330
;     RUN: %{ur-bf} -unroll-count=3 -unroll-uniform-weights | %{fc} UR3330
;     RUN: %{ur-bf} -unroll-count=4 -unroll-uniform-weights | %{fc} UR3330
;     RUN: %{ur-bf} -unroll-count=3 | %{fc} UR3330
;     RUN: %{ur-bf} -unroll-count=4 | %{fc} UR3330
;
;     The new do.body contains 3 of the original loop's iterations, so multiply
;     it by 3 to get the old do.body.
;     ORIG3330: - do.body: float = 3.0,
;     UR3330:   - do.body: float = 1.0,
;
;     UR3330:     call void @f
;     UR3330-NOT: br
;     UR3330:     call void @f
;     UR3330-NOT: br
;     UR3330:     call void @f
;     UR3330:     ret void
;
;   Use a constant iteration count but now the loop upper bound computation can
;   overflow.  When it does, the loop induction variable is greater than it
;   immediately, so the initial unrolled iteration's latch remains conditional.
;
;     RUN: sed -e s/@MAX@/3/ -e s/@W@/2/ -e s/@MIN@/3/ -e s/@I_0@/%x/ %s > %t.ll
;     RUN: %{bf-fc} ORIG333x
;     RUN: %{ur-bf} -unroll-count=3 -unroll-uniform-weights | %{fc} UR333x
;     RUN: %{ur-bf} -unroll-count=4 -unroll-uniform-weights | %{fc} UR333x
;     RUN: %{ur-bf} -unroll-count=3 | %{fc} UR333x
;     RUN: %{ur-bf} -unroll-count=4 | %{fc} UR333x
;
;     The new do.body.1 contains 2 of the original loop's iterations, so
;     multiply it by 2, and add the new do.body to get the old do.body.
;     ORIG333x: - do.body: float = 3.0,
;     UR333x:   - do.body: float = 1.0,
;     UR333x:   - do.body.1: float = 1.0,
;
;     UR333x:     call void @f
;     UR333x: br i1 %{{.*}}, label %do.end, label %do.body.1, !prof !0
;     UR333x:     call void @f
;     UR333x-NOT: br
;     UR333x:     call void @f
;     UR333x:     br label %do.end
;     UR333x:     !0 = !{!"branch_weights", i32 1, i32 2147483647}
;
; Original loop body frequency is 2 (loop weight 1).  This is our first case
; where new frequencies and probabilities are not all approximately 1 or 0.
;
;   First use a variable iteration count so that all non-final unrolled
;   iterations' latches remain conditional.
;
;     RUN: sed -e s/@MAX@/3/ -e s/@W@/1/ -e s/@MIN@/1/ -e s/@I_0@/0/ %s > %t.ll
;     RUN: %{bf-fc} ORIG3210
;     RUN: %{ur-bf} -unroll-count=3 -unroll-uniform-weights | %{fc} UR3210
;     RUN: %{ur-bf} -unroll-count=4 -unroll-uniform-weights | %{fc} UR3210
;     RUN: %{ur-bf} -unroll-count=3 | %{fc} UR3210
;     RUN: %{ur-bf} -unroll-count=4 | %{fc} UR3210
;
;     The sum of the new do.body* is the old do.body.
;     ORIG3210: - do.body: float = 2.0,
;     UR3210:   - do.body: float = 1.0,
;     UR3210:   - do.body.1: float = 0.61803,
;     UR3210:   - do.body.2: float = 0.38197,
;
;     UR3210: call void @f
;     UR3210: br i1 %{{.*}}, label %do.end, label %do.body.1, !prof !0
;     UR3210: call void @f
;     UR3210: br i1 %{{.*}}, label %do.end, label %do.body.2, !prof !0
;     UR3210: call void @f
;     UR3210: br label %do.end
;     UR3210: !0 = !{!"branch_weights", i32 820265763, i32 1327217885}
;
;   Now use a constant iteration count so that all non-final unrolled
;   iterations' latches unconditionally continue.
;
;     RUN: sed -e s/@MAX@/3/ -e s/@W@/1/ -e s/@MIN@/3/ -e s/@I_0@/0/ %s > %t.ll
;     RUN: %{bf-fc} ORIG3230
;     RUN: %{ur-bf} -unroll-count=3 -unroll-uniform-weights | %{fc} UR3230
;     RUN: %{ur-bf} -unroll-count=4 -unroll-uniform-weights | %{fc} UR3230
;     RUN: %{ur-bf} -unroll-count=3 | %{fc} UR3230
;     RUN: %{ur-bf} -unroll-count=4 | %{fc} UR3230
;
;     The new do.body contains 3 of the original loop's iterations, so multiply
;     it by 3, which is greater than the old do.body, which is impossibly low.
;     ORIG3230: - do.body: float = 2.0,
;     UR3230:   - do.body: float = 1.0,
;
;     UR3230:     call void @f
;     UR3230-NOT: br
;     UR3230:     call void @f
;     UR3230-NOT: br
;     UR3230:     call void @f
;     UR3230:     ret void
;
;   Use a constant iteration count but now the loop upper bound computation can
;   overflow.  When it does, the loop induction variable is greater than it
;   immediately, so the initial unrolled iteration's latch remains conditional.
;
;     RUN: sed -e s/@MAX@/3/ -e s/@W@/1/ -e s/@MIN@/3/ -e s/@I_0@/%x/ %s > %t.ll
;     RUN: %{bf-fc} ORIG323x
;     RUN: %{ur-bf} -unroll-count=3 -unroll-uniform-weights | %{fc} UR323x
;     RUN: %{ur-bf} -unroll-count=4 -unroll-uniform-weights | %{fc} UR323x
;     RUN: %{ur-bf} -unroll-count=3 | %{fc} UR323x
;     RUN: %{ur-bf} -unroll-count=4 | %{fc} UR323x
;
;     The new do.body.1 contains 2 of the original loop's iterations, so
;     multiply it by 2, and add the new do.body to get the old do.body.
;     ORIG323x: - do.body: float = 2.0,
;     UR323x:   - do.body: float = 1.0,
;     UR323x:   - do.body.1: float = 0.5,
;
;     UR323x:     call void @f
;     UR323x:     br i1 %{{.*}}, label %do.end, label %do.body.1, !prof !0
;     UR323x:     call void @f
;     UR323x-NOT: br
;     UR323x:     call void @f
;     UR323x:     br label %do.end
;     UR323x:     !0 = !{!"branch_weights", i32 1073741824, i32 1073741824}
;
; Original loop body frequency is 1 (loop weight 0).
;
;   First use a variable iteration count so that all non-final unrolled
;   iterations' latches remain conditional.
;
;     RUN: sed -e s/@MAX@/3/ -e s/@W@/0/ -e s/@MIN@/1/ -e s/@I_0@/0/ %s > %t.ll
;     RUN: %{bf-fc} ORIG3110
;     RUN: %{ur-bf} -unroll-count=3 -unroll-uniform-weights | %{fc} UR3110
;     RUN: %{ur-bf} -unroll-count=4 -unroll-uniform-weights | %{fc} UR3110
;     RUN: %{ur-bf} -unroll-count=3 | %{fc} UR3110
;     RUN: %{ur-bf} -unroll-count=4 | %{fc} UR3110
;
;     The sum of the new do.body* is approximately the old do.body.
;     ORIG3110: - do.body: float = 1.0,
;     UR3110:   - do.body: float = 1.0,
;     UR3110:   - do.body.1: float = 0.0{{(0000[0-9]*)?}},
;     UR3110:   - do.body.2: float = 0.0{{(0000[0-9]*)?}},
;
;     UR3110: call void @f
;     UR3110: br i1 %{{.*}}, label %do.end, label %do.body.1, !prof !0
;     UR3110: call void @f
;     UR3110: br i1 %{{.*}}, label %do.end, label %do.body.2, !prof !0
;     UR3110: call void @f
;     UR3110: br label %do.end
;     UR3110: !0 = !{!"branch_weights", i32 1, i32 0}
;
;   Now use a constant iteration count so that all non-final unrolled
;   iterations' latches unconditionally continue.
;
;     RUN: sed -e s/@MAX@/3/ -e s/@W@/0/ -e s/@MIN@/3/ -e s/@I_0@/0/ %s > %t.ll
;     RUN: %{bf-fc} ORIG3130
;     RUN: %{ur-bf} -unroll-count=3 -unroll-uniform-weights | %{fc} UR3130
;     RUN: %{ur-bf} -unroll-count=4 -unroll-uniform-weights | %{fc} UR3130
;     RUN: %{ur-bf} -unroll-count=3 | %{fc} UR3130
;     RUN: %{ur-bf} -unroll-count=4 | %{fc} UR3130
;
;     The new do.body contains 3 of the original loop's iterations, so multiply
;     it by 3, which is greater than the old do.body, which is impossibly low.
;     ORIG3130: - do.body: float = 1.0,
;     UR3130:   - do.body: float = 1.0,
;
;     UR3130:     call void @f
;     UR3130-NOT: br
;     UR3130:     call void @f
;     UR3130-NOT: br
;     UR3130:     call void @f
;     UR3130:     ret void
;
;   Use a constant iteration count but now the loop upper bound computation can
;   overflow.  When it does, the loop induction variable is greater than it
;   immediately, so the initial unrolled iteration's latch remains conditional.
;
;     RUN: sed -e s/@MAX@/3/ -e s/@W@/0/ -e s/@MIN@/3/ -e s/@I_0@/%x/ %s > %t.ll
;     RUN: %{bf-fc} ORIG313x
;     RUN: %{ur-bf} -unroll-count=3 -unroll-uniform-weights | %{fc} UR313x
;     RUN: %{ur-bf} -unroll-count=4 -unroll-uniform-weights | %{fc} UR313x
;     RUN: %{ur-bf} -unroll-count=3 | %{fc} UR313x
;     RUN: %{ur-bf} -unroll-count=4 | %{fc} UR313x
;
;     The new do.body.1 contains 2 of the original loop's iterations, so
;     multiply it by 2, and add the new do.body to get approximately the old
;     do.body.
;     ORIG313x: - do.body: float = 1.0,
;     UR313x:   - do.body: float = 1.0,
;     UR313x:   - do.body.1: float = 0.0{{(0000[0-9]*)?}},
;
;     UR313x:     call void @f
;     UR313x:     br i1 %{{.*}}, label %do.end, label %do.body.1, !prof !0
;     UR313x:     call void @f
;     UR313x-NOT: br
;     UR313x:     call void @f
;     UR313x:     br label %do.end
;     UR313x:     !0 = !{!"branch_weights", i32 -2147483648, i32 0}

; ------------------------------------------------------------------------------
; Check 4 max iterations:
; - Unroll count of >=4 should always produce complete unrolling.
; - That produces <=3 unrolled iteration latches.  3 is the lowest number where
;   the implementation cannot compute uniform weights using a simple formula.
;   Thus, this is our first case where -unroll-uniform-weights matters.
;
; Original loop body frequency is 5 (loop weight 4), which is impossibly high.
;
;   First use a variable iteration count so that all non-final unrolled
;   iterations' latches remain conditional.
;
;     RUN: sed -e s/@MAX@/4/ -e s/@W@/4/ -e s/@MIN@/1/ -e s/@I_0@/0/ %s > %t.ll
;     RUN: %{bf-fc} ORIG4510
;     RUN: %{ur-bf} -unroll-count=4 -unroll-uniform-weights | %{fc} UR4510
;     RUN: %{ur-bf} -unroll-count=5 -unroll-uniform-weights | %{fc} UR4510
;     RUN: %{ur-bf} -unroll-count=4 | %{fc} UR4510
;     RUN: %{ur-bf} -unroll-count=5 | %{fc} UR4510
;
;     The sum of the new do.body* cannot reach the old do.body, which is
;     impossibly high.
;     ORIG4510: - do.body: float = 5.0,
;     UR4510:   - do.body: float = 1.0,
;     UR4510:   - do.body.1: float = 1.0,
;     UR4510:   - do.body.2: float = 1.0,
;     UR4510:   - do.body.3: float = 1.0,
;
;     The probabilities are maximized to try to reach the original frequency.
;     UR4510: call void @f
;     UR4510: br i1 %{{.*}}, label %do.end, label %do.body.1, !prof !0
;     UR4510: call void @f
;     UR4510: br i1 %{{.*}}, label %do.end, label %do.body.2, !prof !0
;     UR4510: call void @f
;     UR4510: br i1 %{{.*}}, label %do.end, label %do.body.3, !prof !0
;     UR4510: call void @f
;     UR4510: br label %do.end
;     UR4510: !0 = !{!"branch_weights", i32 0, i32 -2147483648}
;
;   Now use a constant iteration count so that all non-final unrolled
;   iterations' latches unconditionally continue.
;
;     RUN: sed -e s/@MAX@/4/ -e s/@W@/4/ -e s/@MIN@/4/ -e s/@I_0@/0/ %s > %t.ll
;     RUN: %{bf-fc} ORIG4540
;     RUN: %{ur-bf} -unroll-count=4 -unroll-uniform-weights | %{fc} UR4540
;     RUN: %{ur-bf} -unroll-count=5 -unroll-uniform-weights | %{fc} UR4540
;     RUN: %{ur-bf} -unroll-count=4 | %{fc} UR4540
;     RUN: %{ur-bf} -unroll-count=5 | %{fc} UR4540
;
;     The new do.body contains 4 of the original loop's iterations, so multiply
;     it by 4, which is less than the old do.body, which is impossibly high.
;     ORIG4540: - do.body: float = 5.0,
;     UR4540:   - do.body: float = 1.0,
;
;     UR4540:     call void @f
;     UR4540-NOT: br
;     UR4540:     call void @f
;     UR4540-NOT: br
;     UR4540:     call void @f
;     UR4540-NOT: br
;     UR4540:     call void @f
;     UR4540:     ret void
;
;   Use a constant iteration count but now the loop upper bound computation can
;   overflow.  When it does, the loop induction variable is greater than it
;   immediately, so the initial unrolled iteration's latch remains conditional.
;
;     RUN: sed -e s/@MAX@/4/ -e s/@W@/4/ -e s/@MIN@/4/ -e s/@I_0@/%x/ %s > %t.ll
;     RUN: %{bf-fc} ORIG454x
;     RUN: %{ur-bf} -unroll-count=4 -unroll-uniform-weights | %{fc} UR454x
;     RUN: %{ur-bf} -unroll-count=5 -unroll-uniform-weights | %{fc} UR454x
;     RUN: %{ur-bf} -unroll-count=4 | %{fc} UR454x
;     RUN: %{ur-bf} -unroll-count=5 | %{fc} UR454x
;
;     The new do.body.1 contains 3 of the original loop's iterations, so
;     multiply it by 3, and add the new do.body, but that sum is less than the
;     old do.body, which is impossibly high.
;     ORIG454x: - do.body: float = 5.0,
;     UR454x:   - do.body: float = 1.0,
;     UR454x:   - do.body.1: float = 1.0,
;
;     The sole probability is maximized to try to reach the original frequency.
;     UR454x:     call void @f
;     UR454x:     br i1 %{{.*}}, label %do.end, label %do.body.1, !prof !0
;     UR454x:     call void @f
;     UR454x-NOT: br
;     UR454x:     call void @f
;     UR454x-NOT: br
;     UR454x:     call void @f
;     UR454x:     br label %do.end
;     UR454x:     !0 = !{!"branch_weights", i32 0, i32 -2147483648}
;
; Original loop body frequency is 4 (loop weight 3).
;
;   First use a variable iteration count so that all non-final unrolled
;   iterations' latches remain conditional.
;
;     RUN: sed -e s/@MAX@/4/ -e s/@W@/3/ -e s/@MIN@/1/ -e s/@I_0@/0/ %s > %t.ll
;     RUN: %{bf-fc} ORIG4410
;     RUN: %{ur-bf} -unroll-count=4 -unroll-uniform-weights | %{fc} UR4410
;     RUN: %{ur-bf} -unroll-count=5 -unroll-uniform-weights | %{fc} UR4410
;     RUN: %{ur-bf} -unroll-count=4 | %{fc} UR4410
;     RUN: %{ur-bf} -unroll-count=5 | %{fc} UR4410
;
;     The sum of the new do.body* is the old do.body.
;     ORIG4410: - do.body: float = 4.0,
;     UR4410:   - do.body: float = 1.0,
;     UR4410:   - do.body.1: float = 1.0,
;     UR4410:   - do.body.2: float = 1.0,
;     UR4410:   - do.body.3: float = 1.0,
;
;     UR4410: call void @f
;     UR4410: br i1 %{{.*}}, label %do.end, label %do.body.1, !prof !0
;     UR4410: call void @f
;     UR4410: br i1 %{{.*}}, label %do.end, label %do.body.2, !prof !0
;     UR4410: call void @f
;     UR4410: br i1 %{{.*}}, label %do.end, label %do.body.3, !prof !0
;     UR4410: call void @f
;     UR4410: br label %do.end
;     UR4410: !0 = !{!"branch_weights", i32 0, i32 -2147483648}
;
;   Now use a constant iteration count so that all non-final unrolled
;   iterations' latches unconditionally continue.
;
;     RUN: sed -e s/@MAX@/4/ -e s/@W@/3/ -e s/@MIN@/4/ -e s/@I_0@/0/ %s > %t.ll
;     RUN: %{bf-fc} ORIG4440
;     RUN: %{ur-bf} -unroll-count=4 -unroll-uniform-weights | %{fc} UR4440
;     RUN: %{ur-bf} -unroll-count=5 -unroll-uniform-weights | %{fc} UR4440
;     RUN: %{ur-bf} -unroll-count=4 | %{fc} UR4440
;     RUN: %{ur-bf} -unroll-count=5 | %{fc} UR4440
;
;     The new do.body contains 4 of the original loop's iterations, so multiply
;     it by 4 to get the old do.body.
;     ORIG4440: - do.body: float = 4.0,
;     UR4440:   - do.body: float = 1.0,
;
;     UR4440:     call void @f
;     UR4440-NOT: br
;     UR4440:     call void @f
;     UR4440-NOT: br
;     UR4440:     call void @f
;     UR4440-NOT: br
;     UR4440:     call void @f
;     UR4440:     ret void
;
;   Use a constant iteration count but now the loop upper bound computation can
;   overflow.  When it does, the loop induction variable is greater than it
;   immediately, so the initial unrolled iteration's latch remains conditional.
;
;     RUN: sed -e s/@MAX@/4/ -e s/@W@/3/ -e s/@MIN@/4/ -e s/@I_0@/%x/ %s > %t.ll
;     RUN: %{bf-fc} ORIG444x
;     RUN: %{ur-bf} -unroll-count=4 -unroll-uniform-weights | %{fc} UR444x
;     RUN: %{ur-bf} -unroll-count=5 -unroll-uniform-weights | %{fc} UR444x
;     RUN: %{ur-bf} -unroll-count=4 | %{fc} UR444x
;     RUN: %{ur-bf} -unroll-count=5 | %{fc} UR444x
;
;     The new do.body.1 contains 3 of the original loop's iterations, so
;     multiply it by 3, and add the new do.body to get the old do.body.
;     ORIG444x: - do.body: float = 4.0,
;     UR444x:   - do.body: float = 1.0,
;     UR444x:   - do.body.1: float = 1.0,
;
;     UR444x:     call void @f
;     UR444x:     br i1 %{{.*}}, label %do.end, label %do.body.1, !prof !0
;     UR444x:     call void @f
;     UR444x-NOT: br
;     UR444x:     call void @f
;     UR444x-NOT: br
;     UR444x:     call void @f
;     UR444x:     br label %do.end
;     UR444x:     !0 = !{!"branch_weights", i32 0, i32 -2147483648}
;
; Original loop body frequency is 3 (loop weight 2).  This is our first case
; where the new probabilities vary (unless -unroll-uniform-weights).
;
;   First use a variable iteration count so that all non-final unrolled
;   iterations' latches remain conditional.
;
;     RUN: sed -e s/@MAX@/4/ -e s/@W@/2/ -e s/@MIN@/1/ -e s/@I_0@/0/ %s > %t.ll
;     RUN: %{bf-fc} ORIG4310
;     RUN: %{ur-bf} -unroll-count=4 -unroll-uniform-weights | %{fc} UR4310,UNIF4310
;     RUN: %{ur-bf} -unroll-count=5 -unroll-uniform-weights | %{fc} UR4310,UNIF4310
;     RUN: %{ur-bf} -unroll-count=4 | %{fc} UR4310,FAST4310
;     RUN: %{ur-bf} -unroll-count=5 | %{fc} UR4310,FAST4310
;
;     The sum of the new do.body* is always approximately the old do.body.
;     ORIG4310: - do.body: float = 3.0,
;     UNIF4310: - do.body: float = 1.0,
;     UNIF4310: - do.body.1: float = 0.81054,
;     UNIF4310: - do.body.2: float = 0.65697,
;     UNIF4310: - do.body.3: float = 0.5325,
;     FAST4310: - do.body: float = 1.0,
;     FAST4310: - do.body.1: float = 0.94737,
;     FAST4310: - do.body.2: float = 0.63158,
;     FAST4310: - do.body.3: float = 0.42105,
;
;     UR4310:        call void @f
;     UR4310:        br i1 %{{.*}}, label %do.end, label %do.body.1, !prof !0
;     UR4310:        call void @f
;     UR4310:        br i1 %{{.*}}, label %do.end, label %do.body.2,
;     UNIF4310-SAME:   !prof !0
;     FAST4310-SAME:   !prof !1
;     UR4310:        call void @f
;     UR4310:        br i1 %{{.*}}, label %do.end, label %do.body.3,
;     UNIF4310-SAME:   !prof !0
;     FAST4310-SAME:   !prof !1
;     UR4310:        call void @f
;     UR4310:        br label %do.end
;     UNIF4310:      !0 = !{!"branch_weights", i32 406871040, i32 1740612608}
;     FAST4310:      !0 = !{!"branch_weights", i32 113025456, i32 2034458192}
;     FAST4310:      !1 = !{!"branch_weights", i32 1, i32 2}
;
;   Now use a constant iteration count so that all non-final unrolled
;   iterations' latches unconditionally continue.
;
;     RUN: sed -e s/@MAX@/4/ -e s/@W@/2/ -e s/@MIN@/4/ -e s/@I_0@/0/ %s > %t.ll
;     RUN: %{bf-fc} ORIG4340
;     RUN: %{ur-bf} -unroll-count=4 -unroll-uniform-weights | %{fc} UR4340
;     RUN: %{ur-bf} -unroll-count=5 -unroll-uniform-weights | %{fc} UR4340
;     RUN: %{ur-bf} -unroll-count=4 | %{fc} UR4340
;     RUN: %{ur-bf} -unroll-count=5 | %{fc} UR4340
;
;     The new do.body contains 4 of the original loop's iterations, so multiply
;     it by 4, which is greater than the old do.body, which is impossibly low.
;     ORIG4340: - do.body: float = 3.0,
;     UR4340:   - do.body: float = 1.0,
;
;     UR4340:     call void @f
;     UR4340-NOT: br
;     UR4340:     call void @f
;     UR4340-NOT: br
;     UR4340:     call void @f
;     UR4340-NOT: br
;     UR4340:     call void @f
;     UR4340:     ret void
;
;   Use a constant iteration count but now the loop upper bound computation can
;   overflow.  When it does, the loop induction variable is greater than it
;   immediately, so the initial unrolled iteration's latch remains conditional.
;
;     RUN: sed -e s/@MAX@/4/ -e s/@W@/2/ -e s/@MIN@/4/ -e s/@I_0@/%x/ %s > %t.ll
;     RUN: %{bf-fc} ORIG434x
;     RUN: %{ur-bf} -unroll-count=4 -unroll-uniform-weights | %{fc} UR434x
;     RUN: %{ur-bf} -unroll-count=5 -unroll-uniform-weights | %{fc} UR434x
;     RUN: %{ur-bf} -unroll-count=4 | %{fc} UR434x
;     RUN: %{ur-bf} -unroll-count=5 | %{fc} UR434x
;
;     The new do.body.1 contains 3 of the original loop's iterations, so
;     multiply it by 3, and add the new do.body to get the old do.body.
;     ORIG434x: - do.body: float = 3.0,
;     UR434x:   - do.body: float = 1.0,
;     UR434x:   - do.body.1: float = 0.66667,
;
;     UR434x:     call void @f
;     UR434x:     br i1 %{{.*}}, label %do.end, label %do.body.1, !prof !0
;     UR434x:     call void @f
;     UR434x-NOT: br
;     UR434x:     call void @f
;     UR434x-NOT: br
;     UR434x:     call void @f
;     UR434x:     br label %do.end
;     UR434x:     !0 = !{!"branch_weights", i32 715827884, i32 1431655764}
;
; Original loop body frequency is 2 (loop weight 1).
;
;   First use a variable iteration count so that all non-final unrolled
;   iterations' latches remain conditional.
;
;     RUN: sed -e s/@MAX@/4/ -e s/@W@/1/ -e s/@MIN@/1/ -e s/@I_0@/0/ %s > %t.ll
;     RUN: %{bf-fc} ORIG4210
;     RUN: %{ur-bf} -unroll-count=4 -unroll-uniform-weights | %{fc} UR4210,UNIF4210
;     RUN: %{ur-bf} -unroll-count=5 -unroll-uniform-weights | %{fc} UR4210,UNIF4210
;     RUN: %{ur-bf} -unroll-count=4 | %{fc} UR4210,FAST4210
;     RUN: %{ur-bf} -unroll-count=5 | %{fc} UR4210,FAST4210
;
;     The sum of the new do.body* is always the old do.body.
;     ORIG4210: - do.body: float = 2.0,
;     UNIF4210: - do.body: float = 1.0,
;     UNIF4210: - do.body.1: float = 0.54369,
;     UNIF4210: - do.body.2: float = 0.2956,
;     UNIF4210: - do.body.3: float = 0.16071,
;     FAST4210: - do.body: float = 1.0,
;     FAST4210: - do.body.1: float = 0.57143,
;     FAST4210: - do.body.2: float = 0.28571,
;     FAST4210: - do.body.3: float = 0.14286,
;
;     UR4210:        call void @f
;     UR4210:        br i1 %{{.*}}, label %do.end, label %do.body.1, !prof !0
;     UR4210:        call void @f
;     UR4210:        br i1 %{{.*}}, label %do.end, label %do.body.2,
;     UNIF4210-SAME:   !prof !0
;     FAST4210-SAME:   !prof !1
;     UR4210:        call void @f
;     UR4210:        br i1 %{{.*}}, label %do.end, label %do.body.3,
;     UNIF4210-SAME:   !prof !0
;     FAST4210-SAME:   !prof !1
;     UR4210:        call void @f
;     UR4210:        br label %do.end
;     UNIF4210:      !0 = !{!"branch_weights", i32 979920896, i32 1167562752}
;     FAST4210:      !0 = !{!"branch_weights", i32 920350135, i32 1227133513}
;     FAST4210:      !1 = !{!"branch_weights", i32 1, i32 1}
;
;   Now use a constant iteration count so that all non-final unrolled
;   iterations' latches unconditionally continue.
;
;     RUN: sed -e s/@MAX@/4/ -e s/@W@/1/ -e s/@MIN@/4/ -e s/@I_0@/0/ %s > %t.ll
;     RUN: %{bf-fc} ORIG4240
;     RUN: %{ur-bf} -unroll-count=4 -unroll-uniform-weights | %{fc} UR4240
;     RUN: %{ur-bf} -unroll-count=5 -unroll-uniform-weights | %{fc} UR4240
;     RUN: %{ur-bf} -unroll-count=4 | %{fc} UR4240
;     RUN: %{ur-bf} -unroll-count=5 | %{fc} UR4240
;
;     The new do.body contains 4 of the original loop's iterations, so multiply
;     it by 4, which is greater than the old do.body, which is impossibly low.
;     ORIG4240: - do.body: float = 2.0,
;     UR4240:   - do.body: float = 1.0,
;
;     UR4240:     call void @f
;     UR4240-NOT: br
;     UR4240:     call void @f
;     UR4240-NOT: br
;     UR4240:     call void @f
;     UR4240-NOT: br
;     UR4240:     call void @f
;     UR4240:     ret void
;
;   Use a constant iteration count but now the loop upper bound computation can
;   overflow.  When it does, the loop induction variable is greater than it
;   immediately, so the initial unrolled iteration's latch remains conditional.
;
;     RUN: sed -e s/@MAX@/4/ -e s/@W@/1/ -e s/@MIN@/4/ -e s/@I_0@/%x/ %s > %t.ll
;     RUN: %{bf-fc} ORIG424x
;     RUN: %{ur-bf} -unroll-count=4 -unroll-uniform-weights | %{fc} UR424x
;     RUN: %{ur-bf} -unroll-count=5 -unroll-uniform-weights | %{fc} UR424x
;     RUN: %{ur-bf} -unroll-count=4 | %{fc} UR424x
;     RUN: %{ur-bf} -unroll-count=5 | %{fc} UR424x
;
;     The new do.body.1 contains 3 of the original loop's iterations, so
;     multiply it by 3, and add the new do.body to get the old do.body.
;     ORIG424x: - do.body: float = 2.0,
;     UR424x:   - do.body: float = 1.0,
;     UR424x:   - do.body.1: float = 0.33333,
;
;     UR424x:     call void @f
;     UR424x:     br i1 %{{.*}}, label %do.end, label %do.body.1, !prof !0
;     UR424x:     call void @f
;     UR424x-NOT: br
;     UR424x:     call void @f
;     UR424x-NOT: br
;     UR424x:     call void @f
;     UR424x:     br label %do.end
;     UR424x:     !0 = !{!"branch_weights", i32 1431655765, i32 715827883}
;
; Original loop body frequency is 1 (loop weight 0).
;
;   First use a variable iteration count so that all non-final unrolled
;   iterations' latches remain conditional.
;
;     RUN: sed -e s/@MAX@/4/ -e s/@W@/0/ -e s/@MIN@/1/ -e s/@I_0@/0/ %s > %t.ll
;     RUN: %{bf-fc} ORIG4110
;     RUN: %{ur-bf} -unroll-count=4 -unroll-uniform-weights | %{fc} UR4110
;     RUN: %{ur-bf} -unroll-count=5 -unroll-uniform-weights | %{fc} UR4110
;     RUN: %{ur-bf} -unroll-count=4 | %{fc} UR4110
;     RUN: %{ur-bf} -unroll-count=5 | %{fc} UR4110
;
;     The sum of the new do.body* is approximately the old do.body.
;     ORIG4110: - do.body: float = 1.0,
;     UR4110:   - do.body: float = 1.0,
;     UR4110:   - do.body.1: float = 0.0{{(0000[0-9]*)?}},
;     UR4110:   - do.body.2: float = 0.0{{(0000[0-9]*)?}},
;     UR4110:   - do.body.3: float = 0.0{{(0000[0-9]*)?}},
;
;     UR4110: call void @f
;     UR4110: br i1 %{{.*}}, label %do.end, label %do.body.1, !prof !0
;     UR4110: call void @f
;     UR4110: br i1 %{{.*}}, label %do.end, label %do.body.2, !prof !0
;     UR4110: call void @f
;     UR4110: br i1 %{{.*}}, label %do.end, label %do.body.3, !prof !0
;     UR4110: call void @f
;     UR4110: br label %do.end
;     UR4110: !0 = !{!"branch_weights", i32 1, i32 0}
;
;   Now use a constant iteration count so that all non-final unrolled
;   iterations' latches unconditionally continue.
;
;     RUN: sed -e s/@MAX@/4/ -e s/@W@/0/ -e s/@MIN@/4/ -e s/@I_0@/0/ %s > %t.ll
;     RUN: %{bf-fc} ORIG4140
;     RUN: %{ur-bf} -unroll-count=4 -unroll-uniform-weights | %{fc} UR4140
;     RUN: %{ur-bf} -unroll-count=5 -unroll-uniform-weights | %{fc} UR4140
;     RUN: %{ur-bf} -unroll-count=4 | %{fc} UR4140
;     RUN: %{ur-bf} -unroll-count=5 | %{fc} UR4140
;
;     The new do.body contains 4 of the original loop's iterations, so multiply
;     it by 4, which is greater than the old do.body, which is impossibly low.
;     ORIG4140: - do.body: float = 1.0,
;     UR4140:   - do.body: float = 1.0,
;
;     UR4140:     call void @f
;     UR4140-NOT: br
;     UR4140:     call void @f
;     UR4140-NOT: br
;     UR4140:     call void @f
;     UR4140-NOT: br
;     UR4140:     call void @f
;     UR4140:     ret void
;
;   Use a constant iteration count but now the loop upper bound computation can
;   overflow.  When it does, the loop induction variable is greater than it
;   immediately, so the initial unrolled iteration's latch remains conditional.
;
;     RUN: sed -e s/@MAX@/4/ -e s/@W@/0/ -e s/@MIN@/4/ -e s/@I_0@/%x/ %s > %t.ll
;     RUN: %{bf-fc} ORIG414x
;     RUN: %{ur-bf} -unroll-count=4 -unroll-uniform-weights | %{fc} UR414x
;     RUN: %{ur-bf} -unroll-count=5 -unroll-uniform-weights | %{fc} UR414x
;     RUN: %{ur-bf} -unroll-count=4 | %{fc} UR414x
;     RUN: %{ur-bf} -unroll-count=5 | %{fc} UR414x
;
;     The new do.body.1 contains 3 of the original loop's iterations, so
;     multiply it by 3, and add the new do.body to get approximately the old
;     do.body.
;     ORIG414x: - do.body: float = 1.0,
;     UR414x:   - do.body: float = 1.0,
;     UR414x:   - do.body.1: float = 0.0{{(0000[0-9]*)?}},
;
;     UR414x:     call void @f
;     UR414x:     br i1 %{{.*}}, label %do.end, label %do.body.1, !prof !0
;     UR414x:     call void @f
;     UR414x-NOT: br
;     UR414x:     call void @f
;     UR414x-NOT: br
;     UR414x:     call void @f
;     UR414x:     br label %do.end
;     UR414x:     !0 = !{!"branch_weights", i32 -2147483648, i32 0}

; ------------------------------------------------------------------------------
; Check 5 max iterations:
; - Unroll count of >=5 should always produce complete unrolling.
; - That produces <=4 unrolled iteration latches.  When at least 3 remain
;   conditional, the implementation cannot compute uniform weights using a
;   simple formula, so -unroll-uniform-weights matters.
;
; Original loop body frequency is 5 (loop weight 4).
;
;   RUN: sed -e s/@MAX@/5/ -e s/@W@/4/ -e s/@MIN@/1/ -e s/@I_0@/0/ %s > %t.ll
;   RUN: %{bf-fc} ORIG5510
;   RUN: %{ur-bf} -unroll-count=5 -unroll-uniform-weights | %{fc} UR5510,UNIF5510
;   RUN: %{ur-bf} -unroll-count=6 -unroll-uniform-weights | %{fc} UR5510,UNIF5510
;   RUN: %{ur-bf} -unroll-count=5 | %{fc} UR5510,FAST5510
;   RUN: %{ur-bf} -unroll-count=6 | %{fc} UR5510,FAST5510
;
;   The sum of the new do.body* is the old do.body.
;   ORIG5510: - do.body: float = 5.0,
;   UR5510:   - do.body: float = 1.0,
;   UR5510:   - do.body.1: float = 1.0,
;   UR5510:   - do.body.2: float = 1.0,
;   UR5510:   - do.body.3: float = 1.0,
;   UR5510:   - do.body.4: float = 1.0,
;
;   All continue probabilities are approximately 1, but somehow there is less
;   precision in the calculation of the last case.
;   UR5510:        call void @f
;   UR5510:        br i1 %{{.*}}, label %do.end, label %do.body.1, !prof !0
;   UR5510:        call void @f
;   UR5510:        br i1 %{{.*}}, label %do.end, label %do.body.2, !prof !0
;   UR5510:        call void @f
;   UR5510:        br i1 %{{.*}}, label %do.end, label %do.body.3, !prof !0
;   UR5510:        call void @f
;   UR5510:        br i1 %{{.*}}, label %do.end, label %do.body.4,
;   UNIF5510-SAME:   !prof !0
;   FAST5510-SAME:   !prof !1
;   UR5510:        call void @f
;   UR5510:        br label %do.end
;   UNIF5510: !0 = !{!"branch_weights", i32 0, i32 -2147483648}
;   FAST5510: !0 = !{!"branch_weights", i32 0, i32 -2147483648}
;   FAST5510: !1 = !{!"branch_weights", i32 10, i32 2147483638}
;
; Original loop body frequency is 4 (loop weight 3).
;
;   RUN: sed -e s/@MAX@/5/ -e s/@W@/3/ -e s/@MIN@/1/ -e s/@I_0@/0/ %s > %t.ll
;   RUN: %{bf-fc} ORIG5410
;   RUN: %{ur-bf} -unroll-count=5 -unroll-uniform-weights | %{fc} UR5410,UNIF5410
;   RUN: %{ur-bf} -unroll-count=6 -unroll-uniform-weights | %{fc} UR5410,UNIF5410
;   RUN: %{ur-bf} -unroll-count=5 | %{fc} UR5410,FAST5410
;   RUN: %{ur-bf} -unroll-count=6 | %{fc} UR5410,FAST5410
;
;   The sum of the new do.body* is always the old do.body.
;   ORIG5410: - do.body: float = 4.0,
;   UNIF5410: - do.body: float = 1.0,
;   UNIF5410: - do.body.1: float = 0.88818,
;   UNIF5410: - do.body.2: float = 0.78886,
;   UNIF5410: - do.body.3: float = 0.70065,
;   UNIF5410: - do.body.4: float = 0.62231,
;   FAST5410: - do.body: float = 1.0,
;   FAST5410: - do.body.1: float = 1.0,
;   FAST5410: - do.body.2: float = 0.86486,
;   FAST5410: - do.body.3: float = 0.64865,
;   FAST5410: - do.body.4: float = 0.48649,
;
;   This is our first case where, when not using -unroll-uniform-weights, the
;   implementation must adjust multiple probabilities to something other than
;   the original latch probability but does not just set all probabilities to
;   the limit of 1 or 0.
;   UR5410:        call void @f
;   UR5410:        br i1 %{{.*}}, label %do.end, label %do.body.1, !prof !0
;   UR5410:        call void @f
;   UR5410:        br i1 %{{.*}}, label %do.end, label %do.body.2,
;   UNIF5410-SAME:   !prof !0
;   FAST5410-SAME:   !prof !1
;   UR5410:        call void @f
;   UR5410:        br i1 %{{.*}}, label %do.end, label %do.body.3,
;   UNIF5410-SAME:   !prof !0
;   FAST5410-SAME:   !prof !2
;   UR5410:        call void @f
;   UR5410:        br i1 %{{.*}}, label %do.end, label %do.body.4,
;   UNIF5410-SAME:   !prof !0
;   FAST5410-SAME:   !prof !2
;   UR5410:        call void @f
;   UR5410:        br label %do.end
;   UNIF5410: !0 = !{!"branch_weights", i32 240132096, i32 1907351552}
;   FAST5410: !0 = !{!"branch_weights", i32 0, i32 -2147483648}
;   FAST5410: !1 = !{!"branch_weights", i32 290200493, i32 1857283155}
;   FAST5410: !2 = !{!"branch_weights", i32 1, i32 3}
;
; Original loop body frequency is 1 (loop weight 0).
;
;   RUN: sed -e s/@MAX@/5/ -e s/@W@/0/ -e s/@MIN@/1/ -e s/@I_0@/0/ %s > %t.ll
;   RUN: %{bf-fc} ORIG5110
;   RUN: %{ur-bf} -unroll-count=5 -unroll-uniform-weights | %{fc} UR5110
;   RUN: %{ur-bf} -unroll-count=6 -unroll-uniform-weights | %{fc} UR5110
;   RUN: %{ur-bf} -unroll-count=5 | %{fc} UR5110
;   RUN: %{ur-bf} -unroll-count=6 | %{fc} UR5110
;
;   The sum of the new do.body* is approximately the old do.body.
;   ORIG5110: - do.body: float = 1.0,
;   UR5110:   - do.body: float = 1.0,
;   UR5110:   - do.body.1: float = 0.0{{(0000[0-9]*)?}},
;   UR5110:   - do.body.2: float = 0.0{{(0000[0-9]*)?}},
;   UR5110:   - do.body.3: float = 0.0{{(0000[0-9]*)?}},
;   UR5110:   - do.body.4: float = 0.0{{(0000[0-9]*)?}},
;
;   UR5110: call void @f
;   UR5110: br i1 %{{.*}}, label %do.end, label %do.body.1, !prof !0
;   UR5110: call void @f
;   UR5110: br i1 %{{.*}}, label %do.end, label %do.body.2, !prof !0
;   UR5110: call void @f
;   UR5110: br i1 %{{.*}}, label %do.end, label %do.body.3, !prof !0
;   UR5110: call void @f
;   UR5110: br i1 %{{.*}}, label %do.end, label %do.body.4, !prof !0
;   UR5110: call void @f
;   UR5110: br label %do.end
;   UR5110: !0 = !{!"branch_weights", i32 1, i32 0}

declare void @f(i32)

define void @test(i32 %x, i32 %n) {
entry:
  %n.min = call i32 @llvm.umax.i32(i32 %n, i32 @MIN@)
  %n.minmax = call i32 @llvm.umin.i32(i32 %n.min, i32 @MAX@)
  %i_n = add i32 @I_0@, %n.minmax
  br label %do.body

do.body:
  %i = phi i32 [ @I_0@, %entry ], [ %inc, %do.body ]
  %inc = add i32 %i, 1
  call void @f(i32 %i)
  %c = icmp uge i32 %inc, %i_n
  br i1 %c, label %do.end, label %do.body, !prof !0

do.end:
  ret void
}

; Loop body frequency is @W@ + 1.
!0 = !{!"branch_weights", i32 1, i32 @W@}

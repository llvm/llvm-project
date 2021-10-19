; RUN: opt -S -passes='require<scalar-evolution>,require<lazy-value-info>,loop-mssa(loop-predication)' -debug-pass-manager < %s 2>&1 | FileCheck %s

; NOTE: LazyValueAnalysis is an arbitrary analysis that just isn't preserved by
;       this pass. If after your change this analysis is preserved by the pass,
;       please update this test some other analysis that isn't preserved.

; CHECK: Running analysis: LazyValueAnalysis on drop_a_wc_and_leave_early
; CHECK: Running pass: LoopPredicationPass on Loop at depth 1 containing: %loop<header><exiting>,%guarded<exiting>,%guarded2<latch><exiting>
; CHECK: Invalidating analysis: LazyValueAnalysis on drop_a_wc_and_leave_early
; CHECK: Running analysis: LazyValueAnalysis on drop_a_wc_and_leave
; CHECK: Running pass: LoopPredicationPass on Loop at depth 1 containing: %loop<header><exiting>,%guarded<exiting>,%guarded2<latch><exiting>
; CHECK: Invalidating analysis: LazyValueAnalysis on drop_a_wc_and_leave


; This test makes the pass drop its attempts to optimize the exit condition in
; `%loop` BB by using unanalyzable `%cond_0` as an exit condition.
define i64 @drop_a_wc_and_leave_early(i64 %length, i64 %n, i1 %cond_0, i1 %cond_1) {
; Make sure the pass has only replaced `%wc2` with `true` in the definition of `%wb_cond`.
; CHECK-LABEL: define i64 @drop_a_wc_and_leave_early(i64 %length, i64 %n, i1 %cond_0, i1 %cond_1) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %wc1 = call i1 @llvm.experimental.widenable.condition()
; CHECK-NEXT:   %wc2 = call i1 @llvm.experimental.widenable.condition()
; CHECK-NEXT:   %exiplicit_guard_cond = and i1 %cond_0, %wc1
; CHECK-NEXT:   br i1 %exiplicit_guard_cond, label %loop.preheader, label %deopt, !prof !0
; CHECK:      deopt:
; CHECK-NEXT:   %deoptret = call i64 (...) @llvm.experimental.deoptimize.i64() [ "deopt"() ]
; CHECK-NEXT:   ret i64 %deoptret
; CHECK:      loop.preheader:
; CHECK-NEXT:   br label %loop
; CHECK:      loop:
; CHECK-NEXT:   %i = phi i64 [ %i.next, %guarded2 ], [ 0, %loop.preheader ]
; CHECK-NEXT:   br i1 %cond_0, label %guarded, label %deopt2, !prof !0
; CHECK:      deopt2:
; CHECK-NEXT:   %deoptret2 = call i64 (...) @llvm.experimental.deoptimize.i64() [ "deopt"() ]
; CHECK-NEXT:   ret i64 %deoptret2
; CHECK:      guarded:
; CHECK-NEXT:   %wb_cond = and i1 %cond_1, true
; CHECK-NEXT:   br i1 %wb_cond, label %guarded2, label %deopt3, !prof !0
; CHECK:      deopt3:
; CHECK-NEXT:   %deoptret3 = call i64 (...) @llvm.experimental.deoptimize.i64() [ "deopt"() ]
; CHECK-NEXT:   ret i64 %deoptret3
; CHECK:      guarded2:
; CHECK-NEXT:   %i.next = add nuw i64 %i, 1
; CHECK-NEXT:   %continue = icmp ult i64 %i.next, %n
; CHECK-NEXT:   br i1 %continue, label %loop, label %exit
; CHECK:      exit:
; CHECK-NEXT:   ret i64 0
; CHECK-NEXT: }

entry:
  %wc1 = call i1 @llvm.experimental.widenable.condition()
  %wc2 = call i1 @llvm.experimental.widenable.condition()
  %exiplicit_guard_cond = and i1 %cond_0, %wc1
  br i1 %exiplicit_guard_cond, label %loop.preheader, label %deopt, !prof !0

deopt:
  %deoptret = call i64 (...) @llvm.experimental.deoptimize.i64() [ "deopt"() ]
  ret i64 %deoptret

loop.preheader:
  br label %loop

loop:
  %i = phi i64 [ %i.next, %guarded2 ], [ 0, %loop.preheader ]
  br i1 %cond_0, label %guarded, label %deopt2, !prof !0

deopt2:
  %deoptret2 = call i64 (...) @llvm.experimental.deoptimize.i64() [ "deopt"() ]
  ret i64 %deoptret2

guarded:
  %wb_cond = and i1 %cond_1, %wc2
  br i1 %wb_cond, label %guarded2, label %deopt3, !prof !0

deopt3:
  %deoptret3 = call i64 (...) @llvm.experimental.deoptimize.i64() [ "deopt"() ]
  ret i64 %deoptret3

guarded2:
  %i.next = add nuw i64 %i, 1
  %continue = icmp ult i64 %i.next, %n
  br i1 %continue, label %loop, label %exit

exit:
  ret i64 0
}

; This test makes the pass drop its attempts to optimize the exit condition in
; `%loop` BB by using trivial `false` as an exit condition.
define i64 @drop_a_wc_and_leave(i64 %n, i1 %cond_0, i1 %cond_1) {
; Make sure the pass has only replaced `%wc2` with `true` in the definition of `%wb_cond`.
; CHECK-LABEL: define i64 @drop_a_wc_and_leave(i64 %n, i1 %cond_0, i1 %cond_1) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %wc1 = call i1 @llvm.experimental.widenable.condition()
; CHECK-NEXT:   %wc2 = call i1 @llvm.experimental.widenable.condition()
; CHECK-NEXT:   %exiplicit_guard_cond = and i1 %cond_0, %wc1
; CHECK-NEXT:   br i1 %exiplicit_guard_cond, label %loop.preheader, label %deopt, !prof !0
; CHECK:      deopt:
; CHECK-NEXT:   %deoptret = call i64 (...) @llvm.experimental.deoptimize.i64() [ "deopt"() ]
; CHECK-NEXT:   ret i64 %deoptret
; CHECK:      loop.preheader:
; CHECK-NEXT:   br label %loop
; CHECK:      loop:
; CHECK-NEXT:   %i = phi i64 [ %i.next, %guarded2 ], [ 0, %loop.preheader ]
; CHECK-NEXT:   br i1 false, label %guarded, label %deopt2, !prof !0
; CHECK:      deopt2:
; CHECK-NEXT:   %deoptret2 = call i64 (...) @llvm.experimental.deoptimize.i64() [ "deopt"() ]
; CHECK-NEXT:   ret i64 %deoptret2
; CHECK:      guarded:
; CHECK-NEXT:   %wb_cond = and i1 %cond_1, true
; CHECK-NEXT:   br i1 %wb_cond, label %guarded2, label %deopt3, !prof !0
; CHECK:      deopt3:
; CHECK-NEXT:   %deoptret3 = call i64 (...) @llvm.experimental.deoptimize.i64() [ "deopt"() ]
; CHECK-NEXT:   ret i64 %deoptret3
; CHECK:      guarded2:
; CHECK-NEXT:   %i.next = add nuw i64 %i, 1
; CHECK-NEXT:   %continue = icmp ult i64 %i.next, %n
; CHECK-NEXT:   br i1 %continue, label %loop, label %exit
; CHECK:      exit:
; CHECK-NEXT:   ret i64 0
; CHECK-NEXT: }

entry:
  %wc1 = call i1 @llvm.experimental.widenable.condition()
  %wc2 = call i1 @llvm.experimental.widenable.condition()
  %exiplicit_guard_cond = and i1 %cond_0, %wc1
  br i1 %exiplicit_guard_cond, label %loop.preheader, label %deopt, !prof !0

deopt:
  %deoptret = call i64 (...) @llvm.experimental.deoptimize.i64() [ "deopt"() ]
  ret i64 %deoptret

loop.preheader:
  br label %loop

loop:
  %i = phi i64 [ %i.next, %guarded2 ], [ 0, %loop.preheader ]
  br i1 false, label %guarded, label %deopt2, !prof !0

deopt2:
  %deoptret2 = call i64 (...) @llvm.experimental.deoptimize.i64() [ "deopt"() ]
  ret i64 %deoptret2

guarded:
  %wb_cond = and i1 %cond_1, %wc2
  br i1 %wb_cond, label %guarded2, label %deopt3, !prof !0

deopt3:
  %deoptret3 = call i64 (...) @llvm.experimental.deoptimize.i64() [ "deopt"() ]
  ret i64 %deoptret3

guarded2:
  %i.next = add nuw i64 %i, 1
  %continue = icmp ult i64 %i.next, %n
  br i1 %continue, label %loop, label %exit

exit:
  ret i64 0
}


declare i1 @llvm.experimental.widenable.condition()
declare i64 @llvm.experimental.deoptimize.i64(...)

!0 = !{!"branch_weights", i64 1048576, i64 1}

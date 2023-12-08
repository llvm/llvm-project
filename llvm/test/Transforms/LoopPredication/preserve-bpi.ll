; RUN: opt -mtriple=x86_64 -passes='loop-mssa(loop-predication,licm,simple-loop-unswitch<nontrivial>,loop-simplifycfg)' -debug-pass-manager -debug-only=branch-prob -S < %s 2>&1 | FileCheck %s

; REQUIRES: asserts

; This test is to solely check that we do not run BPI every single time loop
; predication is invoked (since BPI is preserved as part of
; LoopStandardAnalysisResults).
declare void @llvm.experimental.guard(i1, ...)

; CHECK: Running pass: LoopPredicationPass on loop
; CHECK-NEXT: Running pass: LICMPass on loop
; CHECK-NEXT: Running pass: SimpleLoopUnswitchPass on loop
; CHECK-NEXT: Running analysis: OuterAnalysisManagerProxy
; CHECK-NEXT: Running pass: LoopPredicationPass on loop
; CHECK-NEXT: Running pass: LICMPass on loop
; CHECK-NEXT: Running pass: SimpleLoopUnswitchPass on loop
; CHECK-NEXT: Running pass: LoopSimplifyCFGPass on loop

define i32 @unsigned_loop_0_to_n_ult_check(ptr %array, i32 %length, i32 %n) {
entry:
  %tmp5 = icmp eq i32 %n, 0
  br i1 %tmp5, label %exit, label %loop.preheader

loop.preheader:                                   ; preds = %entry
  br label %loop

loop:                                             ; preds = %guarded, %loop.preheader
  %loop.acc = phi i32 [ %loop.acc.next, %guarded ], [ 0, %loop.preheader ]
  %i = phi i32 [ %i.next, %guarded ], [ 0, %loop.preheader ]
  %within.bounds = icmp ult i32 %i, %length
  %widenable_cond = call i1 @llvm.experimental.widenable.condition()
  %exiplicit_guard_cond = and i1 %within.bounds, %widenable_cond
  br i1 %exiplicit_guard_cond, label %guarded, label %deopt, !prof !0

deopt:                                            ; preds = %loop
  %deoptcall = call i32 (...) @llvm.experimental.deoptimize.i32(i32 9) [ "deopt"() ]
  ret i32 %deoptcall

guarded:                                          ; preds = %loop
  %i.i64 = zext i32 %i to i64
  %array.i.ptr = getelementptr inbounds i32, ptr %array, i64 %i.i64
  %array.i = load i32, ptr %array.i.ptr, align 4
  %loop.acc.next = add i32 %loop.acc, %array.i
  %i.next = add nuw i32 %i, 1
  %continue = icmp ult i32 %i.next, %n
  br i1 %continue, label %loop, label %exit, !prof !2

exit:                                             ; preds = %guarded, %entry
  %result = phi i32 [ 0, %entry ], [ %loop.acc.next, %guarded ]
  ret i32 %result
}

declare i32 @llvm.experimental.deoptimize.i32(...)
declare i1 @llvm.experimental.widenable.condition() #0

attributes #0 = { inaccessiblememonly nounwind }

!0 = !{!"branch_weights", i32 1048576, i32 1}
!1 = !{i32 1, i32 -2147483648}
!2 = !{!"branch_weights", i32 1024, i32 1}

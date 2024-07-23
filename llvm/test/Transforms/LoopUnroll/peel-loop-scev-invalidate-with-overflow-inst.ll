; RUN: opt < %s -S  -passes='print<scalar-evolution>,loop-unroll<peeling;full-unroll-max=0>,print<scalar-evolution>' 2>&1 | FileCheck %s
;
; This test ensures that (extractvalue 0 (with-overflow-inst op0, op1))
; is invalidated by LoopPeel when the operands of with-overflow-inst
; are changed.
;
; In the following case, LoopPeel modifies the CFG into another one
; with %bb7 not dominating %bb2 and %bb3 although %extractvalue is
; still the step for the %bb3 loop. %call has been modified and uses
; different operands but the SCEV value for %extractvalue has not been
; invalidated and still refers to %load in its SCEV operands
; (SCEV(%extractvalue) := -2 + -2 * %load).
;
; When LoopUnroll tries to compute the SCEV for the %bb3 Phi, the
; stale data for %extractvalue is used whereas %load is not available
; in %bb3 which is wrong.
;
; for more details and nice pictures: https://github.com/llvm/llvm-project/issues/97586
;
; Although the reason for the bug was in forgetValue, it is still relevant to
; test if LoopPeel invalidates %extractvalue after changing %call.
;
; forgetValue only walks the users, so calling it on the IV Phis does not
; invalidate %extractvalue (thus forgetLoop does not invalidate it too).
; It has to be done by LoopPeel itself.


define void @loop_peeling_smul_with_overflow() {
; before loop-unroll
; CHECK: Classifying expressions for: @loop_peeling_smul_with_overflow
; CHECK: %extractvalue = extractvalue { i32, i1 } %call, 0
; CHECK-NEXT: -->  (-2 + (-2 * %load))
; CHECK: %phi4 = phi i32 [ %add, %bb3 ], [ 0, %bb2 ]
; CHECK-NEXT: -->  {0,+,(-2 + (-2 * %load))}<nuw><nsw><%bb3>
; after loop-unroll
; CHECK: Classifying expressions for: @loop_peeling_smul_with_overflow
; CHECK: %extractvalue = extractvalue { i32, i1 } %call, 0
; CHECK-NEXT: -->  (-2 * %add8.lcssa)
; CHECK: %phi4 = phi i32 [ %add, %bb3 ], [ 0, %bb2 ]
; CHECK-NEXT: -->  {0,+,(-2 * %add8.lcssa)}<nuw><nsw><%bb3>
;
bb:
  br label %bb1

bb1:                                              ; preds = %bb3, %bb
  %phi = phi i32 [ 0, %bb ], [ %phi4, %bb3 ]
  br label %bb5

bb2:                                              ; preds = %bb7
  %call = call { i32, i1 } @llvm.smul.with.overflow.i32(i32 %add8, i32 -2)
  %extractvalue = extractvalue { i32, i1 } %call, 0
  br label %bb3

bb3:                                              ; preds = %bb3, %bb2
  %phi4 = phi i32 [ %add, %bb3 ], [ 0, %bb2 ]
  %add = add i32 %extractvalue, %phi4
  br i1 false, label %bb3, label %bb1

bb5:                                              ; preds = %bb7, %bb1
  %phi6 = phi i32 [ 1, %bb1 ], [ 0, %bb7 ]
  %icmp = icmp eq i32 %phi, 0
  br i1 %icmp, label %bb9, label %bb7

bb7:                                              ; preds = %bb5
  %load = load i32, ptr addrspace(1) null, align 4
  %add8 = add i32 %load, 1
  br i1 false, label %bb2, label %bb5

bb9:                                              ; preds = %bb5
  ret void
}

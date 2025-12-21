; Remove 'S' Scalar Dependencies #119345
; Scalar dependencies are not handled correctly, so they were removed to avoid
; miscompiles. The loop nest in this test case used to be interchanged, but it's
; no longer triggering. XFAIL'ing this test to indicate that this test should
; interchanged if scalar deps are handled correctly.
;
; XFAIL: *

; RUN: opt -passes=loop-interchange -cache-line-size=64 -loop-interchange-threshold=-10 %s -pass-remarks-output=%t -disable-output
; RUN: FileCheck -input-file %t %s

; The test contains a GEP with an operand that is not SCEV-able. Make sure
; loop-interchange does not crash.
;
; CHECK:       --- !Passed
; CHECK-NEXT:  Pass:            loop-interchange
; CHECK-NEXT:  Name:            Interchanged
; CHECK-NEXT:  Function:        test
; CHECK-NEXT:  Args:
; CHECK-NEXT:    - String:          Loop interchanged with enclosing loop.

define void @test(ptr noalias %src, ptr %dst) {
entry:
  br label %outer.header

outer.header:
  %i = phi i32 [ %i.next, %outer.latch ], [ 0, %entry ]
  br label %inner

inner:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner ]
  %src.gep = getelementptr inbounds [256 x float], ptr %src, <2 x i64> <i64 0, i64 1>, i64 %j
  %src.0 = extractelement <2 x ptr> %src.gep, i32 0
  %lv.0 = load float, ptr %src.0
  %add.0 = fadd float %lv.0, 1.0
  %dst.gep = getelementptr inbounds float, ptr %dst, i64 %j
  store float %add.0, ptr %dst.gep
  %j.next = add nuw nsw i64 %j, 1
  %inner.exitcond = icmp eq i64 %j.next, 100
  br i1 %inner.exitcond, label %outer.latch, label %inner

outer.latch:
  %i.next = add nuw nsw i32 %i, 1
  %outer.exitcond = icmp eq i32 %i.next, 100
  br i1 %outer.exitcond, label %exit, label %outer.header

exit:
  ret void
}

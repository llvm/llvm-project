; Remove 'S' Scalar Dependencies #119345
; Scalar dependencies are not handled correctly, so they were removed to avoid
; miscompiles. The loop nest in this test case used to be interchanged, but it's
; no longer triggering. XFAIL'ing this test to indicate that this test should
; interchanged if scalar deps are handled correctly.
;
; XFAIL: *

; RUN: opt -passes=loop-interchange -cache-line-size=64 -verify-loop-lcssa %s -pass-remarks-output=%t -disable-output
; RUN: FileCheck -input-file %t %s

; Tests for PR43797.

@wdtdr = external dso_local global [5 x [5 x double]], align 16

; CHECK:       --- !Passed
; CHECK-NEXT:  Pass:            loop-interchange
; CHECK-NEXT:  Name:            Interchanged
; CHECK-NEXT:  Function:        test1
; CHECK-NEXT:  Args:
; CHECK-NEXT:    - String:          Loop interchanged with enclosing loop.
; CHECK-NEXT:  ...

define void @test1() {
entry:
  br label %outer.header

outer.header:                                         ; preds = %for.inc27, %entry
  %outer.idx = phi i64 [ 0, %entry ], [ %outer.idx.inc, %outer.latch ]
  %arrayidx8 = getelementptr inbounds [5 x [5 x double]], ptr @wdtdr, i64 0, i64 0, i64 %outer.idx
  br label %inner.header

inner.header:                                        ; preds = %for.inc, %for.body
  %inner.idx = phi i64 [ 0, %outer.header ], [ %inner.idx.inc, %inner.latch]
  %0 = load double, ptr %arrayidx8, align 8
  store double undef, ptr %arrayidx8, align 8
  br label %inner.latch

inner.latch:                                          ; preds = %for.body6
  %inner.idx.inc = add nsw i64 %inner.idx, 1
  br i1 false, label %inner.header, label %inner.exit

inner.exit:                                          ; preds = %for.inc
  %outer.v = add nsw i64 %outer.idx, 1
  br label %outer.latch

outer.latch:                                        ; preds = %for.end
  %outer.idx.inc = add nsw i64 %outer.idx, 1
  br i1 false, label %outer.header, label %outer.exit

outer.exit:                                        ; preds = %for.inc27
  %exit1.lcssa = phi i64 [ %outer.v, %outer.latch ]
  %exit2.lcssa = phi i64 [ %outer.idx.inc, %outer.latch ]
  ret void
}

; CHECK:       --- !Passed
; CHECK-NEXT:  Pass:            loop-interchange
; CHECK-NEXT:  Name:            Interchanged
; CHECK-NEXT:  Function:        test2
; CHECK-NEXT:  Args:
; CHECK-NEXT:    - String:          Loop interchanged with enclosing loop.
; CHECK-NEXT:  ...

define void @test2(i1 %cond) {
entry:
  br i1 %cond, label %outer.header, label %outer.exit

outer.header:                                         ; preds = %for.inc27, %entry
  %outer.idx = phi i64 [ 0, %entry ], [ %outer.idx.inc, %outer.latch ]
  %arrayidx8 = getelementptr inbounds [5 x [5 x double]], ptr @wdtdr, i64 0, i64 0, i64 %outer.idx
  br label %inner.header

inner.header:                                        ; preds = %for.inc, %for.body
  %inner.idx = phi i64 [ 0, %outer.header ], [ %inner.idx.inc, %inner.latch]
  %0 = load double, ptr %arrayidx8, align 8
  store double undef, ptr %arrayidx8, align 8
  br label %inner.latch

inner.latch:                                          ; preds = %for.body6
  %inner.idx.inc = add nsw i64 %inner.idx , 1
  br i1 false, label %inner.header, label %inner.exit

inner.exit:                                          ; preds = %for.inc
  %outer.v = add nsw i64 %outer.idx, 1
  br label %outer.latch

outer.latch:                                        ; preds = %for.end
  %outer.idx.inc = add nsw i64 %outer.idx, 1
  br i1 false, label %outer.header, label %outer.exit

outer.exit:                                        ; preds = %for.inc27
  %exit1.lcssa = phi i64 [ 0, %entry ], [ %outer.v, %outer.latch ]
  %exit2.lcssa = phi i64 [ 0, %entry ], [ %outer.idx.inc, %outer.latch ]
  ret void
}

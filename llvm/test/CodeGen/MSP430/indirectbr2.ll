; RUN: llc -march=msp430 < %s | FileCheck %s
@C.0.2070 = private constant [5 x ptr] [ptr blockaddress(@foo, %L1), ptr blockaddress(@foo, %L2), ptr blockaddress(@foo, %L3), ptr blockaddress(@foo, %L4), ptr blockaddress(@foo, %L5)] ; <[5 x i8*]*> [#uses=1]

define internal i16 @foo(i16 %i) nounwind {
entry:
  %tmp1 = getelementptr inbounds [5 x ptr], ptr @C.0.2070, i16 0, i16 %i ; <i8**> [#uses=1]
  %gotovar.4.0 = load ptr, ptr %tmp1, align 4        ; <i8*> [#uses=1]
; CHECK: br .LC.0.2070(r12)
  indirectbr ptr %gotovar.4.0, [label %L5, label %L4, label %L3, label %L2, label %L1]

L5:                                               ; preds = %bb2
  br label %L4

L4:                                               ; preds = %L5, %bb2
  %res.0 = phi i16 [ 385, %L5 ], [ 35, %entry ]     ; <i16> [#uses=1]
  br label %L3

L3:                                               ; preds = %L4, %bb2
  %res.1 = phi i16 [ %res.0, %L4 ], [ 5, %entry ]   ; <i16> [#uses=1]
  br label %L2

L2:                                               ; preds = %L3, %bb2
  %res.2 = phi i16 [ %res.1, %L3 ], [ 1, %entry ]   ; <i16> [#uses=1]
  br label %L1

L1:                                               ; preds = %L2, %bb2
  %res.3 = phi i16 [ %res.2, %L2 ], [ 2, %entry ]  ; <i16> [#uses=1]
  ret i16 %res.3
}

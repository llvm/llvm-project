; RUN: llc -march=msp430 < %s

@nextaddr = global i8* null                       ; <i8**> [#uses=2]
@C.0.2070 = private constant [5 x i8*] [i8* blockaddress(@foo, %L1), i8* blockaddress(@foo, %L2), i8* blockaddress(@foo, %L3), i8* blockaddress(@foo, %L4), i8* blockaddress(@foo, %L5)] ; <[5 x i8*]*> [#uses=1]

define internal i16 @foo(i16 %i) nounwind {
entry:
  %0 = load i8*, i8** @nextaddr, align 4               ; <i8*> [#uses=2]
  %1 = icmp eq i8* %0, null                       ; <i1> [#uses=1]
  br i1 %1, label %bb3, label %bb2

bb2:                                              ; preds = %bb3, %entry
  %gotovar.4.0 = phi i8* [ %gotovar.4.0.pre, %bb3 ], [ %0, %entry ] ; <i8*> [#uses=1]
  indirectbr i8* %gotovar.4.0, [label %L5, label %L4, label %L3, label %L2, label %L1]

bb3:                                              ; preds = %entry
  %2 = getelementptr inbounds [5 x i8*], [5 x i8*]* @C.0.2070, i16 0, i16 %i ; <i8**> [#uses=1]
  %gotovar.4.0.pre = load i8*, i8** %2, align 4        ; <i8*> [#uses=1]
  br label %bb2

L5:                                               ; preds = %bb2
  br label %L4

L4:                                               ; preds = %L5, %bb2
  %res.0 = phi i16 [ 385, %L5 ], [ 35, %bb2 ]     ; <i16> [#uses=1]
  br label %L3

L3:                                               ; preds = %L4, %bb2
  %res.1 = phi i16 [ %res.0, %L4 ], [ 5, %bb2 ]   ; <i16> [#uses=1]
  br label %L2

L2:                                               ; preds = %L3, %bb2
  %res.2 = phi i16 [ %res.1, %L3 ], [ 1, %bb2 ]   ; <i16> [#uses=1]
  %phitmp = mul i16 %res.2, 6                     ; <i16> [#uses=1]
  br label %L1

L1:                                               ; preds = %L2, %bb2
  %res.3 = phi i16 [ %phitmp, %L2 ], [ 2, %bb2 ]  ; <i16> [#uses=1]
  store i8* blockaddress(@foo, %L5), i8** @nextaddr, align 4
  ret i16 %res.3
}

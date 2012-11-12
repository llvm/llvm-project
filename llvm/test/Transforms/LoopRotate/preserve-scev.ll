; RUN: opt < %s -loop-rotate -loop-reduce -verify-dom-info -verify-loop-info -disable-output

define fastcc void @foo() nounwind {
BB:
  br label %BB1

BB1:                                              ; preds = %BB19, %BB
  br label %BB4

BB2:                                              ; preds = %BB4
  %tmp = bitcast i32 undef to i32                 ; <i32> [#uses=1]
  br label %BB4

BB4:                                              ; preds = %BB3, %BB1
  %tmp5 = phi i32 [ undef, %BB1 ], [ %tmp, %BB2 ] ; <i32> [#uses=1]
  br i1 false, label %BB8, label %BB2

BB8:                                              ; preds = %BB6
  %tmp7 = bitcast i32 %tmp5 to i32                ; <i32> [#uses=2]
  br i1 false, label %BB9, label %BB13

BB9:                                              ; preds = %BB12, %BB8
  %tmp10 = phi i32 [ %tmp11, %BB12 ], [ %tmp7, %BB8 ] ; <i32> [#uses=2]
  %tmp11 = add i32 %tmp10, 1                      ; <i32> [#uses=1]
  br label %BB12

BB12:                                             ; preds = %BB9
  br i1 false, label %BB9, label %BB17

BB13:                                             ; preds = %BB15, %BB8
  %tmp14 = phi i32 [ %tmp16, %BB15 ], [ %tmp7, %BB8 ] ; <i32> [#uses=1]
  br label %BB15

BB15:                                             ; preds = %BB13
  %tmp16 = add i32 %tmp14, -1                     ; <i32> [#uses=1]
  br i1 false, label %BB13, label %BB18

BB17:                                             ; preds = %BB12
  br label %BB19

BB18:                                             ; preds = %BB15
  br label %BB19

BB19:                                             ; preds = %BB18, %BB17
  %tmp20 = phi i32 [ %tmp10, %BB17 ], [ undef, %BB18 ] ; <i32> [#uses=0]
  br label %BB1
}

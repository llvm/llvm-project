; RUN: opt < %s -loop-rotate -loop-reduce -verify-memoryssa -verify-dom-info -verify-loop-info -disable-output

define fastcc void @foo(i32* %A, i64 %i) nounwind {
BB:
  br label %BB1

BB1:                                              ; preds = %BB19, %BB
  %tttmp1 = getelementptr i32, i32* %A, i64 %i
  %tttmp2 = load i32, i32* %tttmp1
  %tttmp3 = add i32 %tttmp2, 1
  store i32 %tttmp3, i32* %tttmp1
  br label %BB4

BB2:                                              ; preds = %BB4
  %tmp = bitcast i32 undef to i32                 ; <i32> [#uses=1]
  %tttmp7 = getelementptr i32, i32* %A, i64 %i
  %tttmp8 = load i32, i32* %tttmp7
  %tttmp9 = add i32 %tttmp8, 3
  store i32 %tttmp9, i32* %tttmp7
  br label %BB4

BB4:                                              ; preds = %BB2, %BB1
  %tmp5 = phi i32 [ undef, %BB1 ], [ %tmp, %BB2 ] ; <i32> [#uses=1]
  %tttmp4 = getelementptr i32, i32* %A, i64 %i
  %tttmp5 = load i32, i32* %tttmp4
  %tttmp6 = add i32 %tttmp5, 3
  store i32 %tttmp6, i32* %tttmp4
  br i1 false, label %BB8, label %BB2

BB8:                                              ; preds = %BB6
  %tmp7 = bitcast i32 %tmp5 to i32                ; <i32> [#uses=2]
  %tttmp10 = getelementptr i32, i32* %A, i64 %i
  %tttmp11 = load i32, i32* %tttmp10
  %tttmp12 = add i32 %tttmp11, 3
  store i32 %tttmp12, i32* %tttmp10
  br i1 false, label %BB9, label %BB13

BB9:                                              ; preds = %BB12, %BB8
  %tmp10 = phi i32 [ %tmp11, %BB12 ], [ %tmp7, %BB8 ] ; <i32> [#uses=2]
  %tmp11 = add i32 %tmp10, 1                      ; <i32> [#uses=1]
  %tttmp13 = getelementptr i32, i32* %A, i64 %i
  %tttmp14 = load i32, i32* %tttmp13
  %tttmp15 = add i32 %tttmp14, 3
  store i32 %tttmp15, i32* %tttmp13
  br label %BB12

BB12:                                             ; preds = %BB9
  br i1 false, label %BB9, label %BB17

BB13:                                             ; preds = %BB15, %BB8
  %tmp14 = phi i32 [ %tmp16, %BB15 ], [ %tmp7, %BB8 ] ; <i32> [#uses=1]
  %tttmp16 = getelementptr i32, i32* %A, i64 %i
  %tttmp17 = load i32, i32* %tttmp16
  %tttmp18 = add i32 %tttmp17, 3
  store i32 %tttmp18, i32* %tttmp16
  br label %BB15

BB15:                                             ; preds = %BB13
  %tmp16 = add i32 %tmp14, -1                     ; <i32> [#uses=1]
  %tttmp19 = getelementptr i32, i32* %A, i64 %i
  %tttmp20 = load i32, i32* %tttmp19
  %tttmp21 = add i32 %tttmp20, 3
  store i32 %tttmp21, i32* %tttmp19
  br i1 false, label %BB13, label %BB18

BB17:                                             ; preds = %BB12
  br label %BB19

BB18:                                             ; preds = %BB15
  %tttmp22 = getelementptr i32, i32* %A, i64 %i
  %tttmp23 = load i32, i32* %tttmp22
  %tttmp24 = add i32 %tttmp23, 3
  store i32 %tttmp24, i32* %tttmp22
  br label %BB19

BB19:                                             ; preds = %BB18, %BB17
  %tmp20 = phi i32 [ %tmp10, %BB17 ], [ undef, %BB18 ] ; <i32> [#uses=0]
  br label %BB1
}

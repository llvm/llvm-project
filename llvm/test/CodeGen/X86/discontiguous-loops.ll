; RUN: llc -verify-loop-info -verify-dom-info -march=x86-64 < %s
; PR5243

@.str96 = external constant [37 x i8], align 8    ; <[37 x i8]*> [#uses=1]

define void @foo() nounwind {
bb:
  br label %ybb1

ybb1:                                              ; preds = %yybb13, %xbb6, %bb
  switch i32 undef, label %bb18 [
    i32 150, label %ybb2
    i32 151, label %bb17
    i32 152, label %bb19
    i32 157, label %ybb8
  ]

ybb2:                                              ; preds = %ybb1
  %tmp = icmp eq i8** undef, null                 ; <i1> [#uses=1]
  br i1 %tmp, label %bb3, label %xbb6

bb3:                                              ; preds = %ybb2
  unreachable

xbb4:                                              ; preds = %xbb6
  store i32 0, i32* undef, align 8
  br i1 undef, label %xbb6, label %bb5

bb5:                                              ; preds = %xbb4
  call fastcc void @decl_mode_check_failed() nounwind
  unreachable

xbb6:                                              ; preds = %xbb4, %ybb2
  %tmp7 = icmp slt i32 undef, 0                   ; <i1> [#uses=1]
  br i1 %tmp7, label %xbb4, label %ybb1

ybb8:                                              ; preds = %ybb1
  %tmp9 = icmp eq i8** undef, null                ; <i1> [#uses=1]
  br i1 %tmp9, label %bb10, label %ybb12

bb10:                                             ; preds = %ybb8
  %tmp11 = load i8*, i8** undef, align 8               ; <i8*> [#uses=1]
  call void (i8*, ...) @fatal(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str96, i64 0, i64 0), i8* %tmp11) nounwind
  unreachable

ybb12:                                             ; preds = %ybb8
  br i1 undef, label %bb15, label %ybb13

ybb13:                                             ; preds = %ybb12
  %tmp14 = icmp sgt i32 undef, 0                  ; <i1> [#uses=1]
  br i1 %tmp14, label %bb16, label %ybb1

bb15:                                             ; preds = %ybb12
  call void (i8*, ...) @fatal(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str96, i64 0, i64 0), i8* undef) nounwind
  unreachable

bb16:                                             ; preds = %ybb13
  unreachable

bb17:                                             ; preds = %ybb1
  unreachable

bb18:                                             ; preds = %ybb1
  unreachable

bb19:                                             ; preds = %ybb1
  unreachable
}

declare void @fatal(i8*, ...)

declare fastcc void @decl_mode_check_failed() nounwind

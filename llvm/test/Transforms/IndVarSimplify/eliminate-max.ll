; RUN: opt < %s -S -indvars | grep "= icmp" | count 3
; PR4914.ll

; Indvars should be able to do range analysis and eliminate icmps.
; There are two here which cannot be eliminated.
; There's one that icmp which can be eliminated and which indvars currently
; cannot eliminate, because it requires analyzing more than just the
; range of the induction variable.

@0 = private constant [4 x i8] c"%d\0A\00", align 1 ; <[4 x i8]*> [#uses=1]

define i32 @main() nounwind {
bb:
  br label %bb1

bb1:                                              ; preds = %bb14, %bb
  %t = phi i32 [ 0, %bb ], [ %t19, %bb14 ]        ; <i32> [#uses=5]
  %t2 = phi i32 [ 0, %bb ], [ %t18, %bb14 ]       ; <i32> [#uses=1]
  %t3 = icmp slt i32 %t, 0                        ; <i1> [#uses=1]
  br i1 %t3, label %bb7, label %bb4

bb4:                                              ; preds = %bb1
  %t5 = icmp sgt i32 %t, 255                      ; <i1> [#uses=1]
  %t6 = select i1 %t5, i32 255, i32 %t            ; <i32> [#uses=1]
  br label %bb7

bb7:                                              ; preds = %bb4, %bb1
  %t8 = phi i32 [ %t6, %bb4 ], [ 0, %bb1 ]        ; <i32> [#uses=1]
  %t9 = sub i32 0, %t                             ; <i32> [#uses=3]
  %t10 = icmp slt i32 %t9, 0                      ; <i1> [#uses=1]
  br i1 %t10, label %bb14, label %bb11

bb11:                                             ; preds = %bb7
  %t12 = icmp sgt i32 %t9, 255                    ; <i1> [#uses=1]
  %t13 = select i1 %t12, i32 255, i32 %t9         ; <i32> [#uses=1]
  br label %bb14

bb14:                                             ; preds = %bb11, %bb7
  %t15 = phi i32 [ %t13, %bb11 ], [ 0, %bb7 ]     ; <i32> [#uses=1]
  %t16 = add nsw i32 %t2, 255                     ; <i32> [#uses=1]
  %t17 = add nsw i32 %t16, %t8                    ; <i32> [#uses=1]
  %t18 = add nsw i32 %t17, %t15                   ; <i32> [#uses=2]
  %t19 = add nsw i32 %t, 1                        ; <i32> [#uses=2]
  %t20 = icmp slt i32 %t19, 1000000000            ; <i1> [#uses=1]
  br i1 %t20, label %bb1, label %bb21

bb21:                                             ; preds = %bb14
  %t22 = call i32 (i8*, ...)* @printf(i8* noalias getelementptr inbounds ([4 x i8]* @0, i32 0, i32 0), i32 %t18) nounwind
  ret i32 0
}

declare i32 @printf(i8* noalias nocapture, ...) nounwind

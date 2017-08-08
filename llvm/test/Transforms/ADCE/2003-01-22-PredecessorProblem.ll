; Testcase reduced from 197.parser by bugpoint
; RUN: opt < %s -adce 
; RUN: opt < %s -adce -adce-remove-loops -S | FileCheck %s

define void @conjunction_prune() {
; <label>:0
        br label %bb19

bb19:           ; preds = %bb23, %bb22, %0
        %reg205 = phi i8* [ null, %bb22 ], [ null, %bb23 ], [ null, %0 ]                ; <i8*> [#uses=1]
; CHECK: br label %bb22
        br i1 false, label %bb21, label %bb22

bb21:           ; preds = %bb19
        %cast455 = bitcast i8* %reg205 to i8**          ; <i8**> [#uses=0]
; CHECK: br label %bb22
        br label %bb22

bb22:           ; preds = %bb21, %bb19
; CHECK: br label %bb23
        br i1 false, label %bb19, label %bb23

bb23:           ; preds = %bb22
; CHECK: br label %bb28
        br i1 false, label %bb19, label %bb28

bb28:           ; preds = %bb23
        ret void
}


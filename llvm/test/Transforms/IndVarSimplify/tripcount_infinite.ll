; These tests have an infinite trip count.  We obviously shouldn't remove the 
; loops!  :)
;
; RUN: opt < %s -indvars -adce -simplifycfg -S | FileCheck %s

;; test for (i = 1; i != 100; i += 2)
define i32 @infinite_linear() {
; CHECK-LABEL: @infinite_linear(
entry:
        br label %loop

loop:           ; preds = %loop, %entry
; CHECK-LABEL: loop:
        %i = phi i32 [ 1, %entry ], [ %i.next, %loop ]          ; <i32> [#uses=3]
        %i.next = add i32 %i, 2         ; <i32> [#uses=1]
        %c = icmp ne i32 %i, 100                ; <i1> [#uses=1]
; CHECK: icmp
; CHECK: br
        br i1 %c, label %loop, label %loopexit

loopexit:               ; preds = %loop
; CHECK-LABEL: loopexit:
        ret i32 %i
}

;; test for (i = 1; i*i != 63; ++i)
define i32 @infinite_quadratic() {
; CHECK-LABEL: @infinite_quadratic(
entry:
        br label %loop

loop:           ; preds = %loop, %entry
; CHECK-LABEL: loop:
        %i = phi i32 [ 1, %entry ], [ %i.next, %loop ]          ; <i32> [#uses=4]
        %isquare = mul i32 %i, %i               ; <i32> [#uses=1]
        %i.next = add i32 %i, 1         ; <i32> [#uses=1]
        %c = icmp ne i32 %isquare, 63           ; <i1> [#uses=1]
; CHECK: icmp
; CHECK: br
        br i1 %c, label %loop, label %loopexit

loopexit:               ; preds = %loop
; CHECK-LABEL: loopexit:
        ret i32 %i
}

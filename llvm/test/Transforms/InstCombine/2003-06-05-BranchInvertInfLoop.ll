; This testcase causes an infinite loop in the instruction combiner,
; because it things that the constant value is a not expression... and 
; constantly inverts the branch back and forth.
;
; RUN: opt < %s -instcombine -disable-output

define i8 @test19(i1 %c) {
        br i1 true, label %True, label %False

True:           ; preds = %0
        ret i8 1

False:          ; preds = %0
        ret i8 3
}


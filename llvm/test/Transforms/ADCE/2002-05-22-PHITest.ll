; It is illegal to remove BB1 because it will mess up the PHI node!
;
; RUN: opt < %s -adce -S | grep BB1

define i32 @test(i1 %C, i32 %A, i32 %B) {
; <label>:0
        br i1 %C, label %BB1, label %BB2

BB1:            ; preds = %0
        br label %BB2

BB2:            ; preds = %BB1, %0
        %R = phi i32 [ %A, %0 ], [ %B, %BB1 ]           ; <i32> [#uses=1]
        ret i32 %R
}


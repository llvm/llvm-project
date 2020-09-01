; This is a basic sanity check for constant propagation.  The add instruction
; should be eliminated.

; RUN: opt < %s -instsimplify -die -S | not grep phi

define i32 @test(i1 %B) {
BB0:
        br i1 %B, label %BB1, label %BB3

BB1:            ; preds = %BB0
        br label %BB3

BB3:            ; preds = %BB1, %BB0
        %Ret = phi i32 [ 1, %BB0 ], [ 1, %BB1 ]         ; <i32> [#uses=1]
        ret i32 %Ret
}


; The unify-function-exit-nodes pass often makes basic blocks that just contain
; a PHI node and a return.  Make sure the simplify cfg can straighten out this
; important case.  This is basically the most trivial form of tail-duplication.

; RUN: opt < %s -simplifycfg -S | \
; RUN:    not grep "br label"

define i32 @test(i1 %B, i32 %A, i32 %B.upgrd.1) {
        br i1 %B, label %T, label %F
T:              ; preds = %0
        br label %ret
F:              ; preds = %0
        br label %ret
ret:            ; preds = %F, %T
        %X = phi i32 [ %A, %F ], [ %B.upgrd.1, %T ]             ; <i32> [#uses=1]
        ret i32 %X
}


; Make sure it's willing to move unconditional branches to return instructions
; as well, even if the return block is shared and the source blocks are
; non-empty.
define i32 @test2(i1 %B, i32 %A, i32 %B.upgrd.2) {
        br i1 %B, label %T, label %F
T:              ; preds = %0
        call i32 @test( i1 true, i32 5, i32 8 )         ; <i32>:1 [#uses=0]
        br label %ret
F:              ; preds = %0
        call i32 @test( i1 true, i32 5, i32 8 )         ; <i32>:2 [#uses=0]
        br label %ret
ret:            ; preds = %F, %T
        ret i32 %A
}

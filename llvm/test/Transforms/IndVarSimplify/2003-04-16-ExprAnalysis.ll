; This is a test case for the expression analysis code, not really indvars.
; It was assuming any constant of int type was a ConstantInteger.
;
; RUN: opt < %s -indvars

@X = global i32 7               ; <i32*> [#uses=1]

define void @test(i32 %A) {
; <label>:0
        br label %Loop

Loop:           ; preds = %Loop, %0
        %IV = phi i32 [ %A, %0 ], [ %IVNext, %Loop ]            ; <i32> [#uses=1]
        %IVNext = add i32 %IV, ptrtoint (i32* @X to i32)                ; <i32> [#uses=1]
        br label %Loop
}


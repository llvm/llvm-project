; Induction variable pass is doing bad things with pointer induction vars, 
; trying to do arithmetic on them directly.
;
; RUN: opt < %s -passes=indvars
;
define void @test(i32 %A, i32 %S, ptr %S.upgrd.1) {
; <label>:0
        br label %Loop

Loop:           ; preds = %Loop, %0
        %PIV = phi ptr [ %S.upgrd.1, %0 ], [ %PIVNext.upgrd.3, %Loop ]          ; <ptr> [#uses=1]
        %PIV.upgrd.2 = ptrtoint ptr %PIV to i64         ; <i64> [#uses=1]
        %PIVNext = add i64 %PIV.upgrd.2, 8              ; <i64> [#uses=1]
        %PIVNext.upgrd.3 = inttoptr i64 %PIVNext to ptr         ; <ptr> [#uses=1]
        br label %Loop
}


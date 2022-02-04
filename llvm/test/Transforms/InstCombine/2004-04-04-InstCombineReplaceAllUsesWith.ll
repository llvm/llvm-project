; RUN: opt < %s -passes=instcombine -disable-output

define i32 @test() {
        ret i32 0

Loop:           ; preds = %Loop
        %X = add i32 %X, 1              ; <i32> [#uses=1]
        br label %Loop
}


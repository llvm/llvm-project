; RUN: opt < %s -instcombine -disable-output

; This example caused instcombine to spin into an infinite loop.

define void @test(i32* %P) {
        ret void

Dead:           ; preds = %Dead
        %X = phi i32 [ %Y, %Dead ]              ; <i32> [#uses=1]
        %Y = sdiv i32 %X, 10            ; <i32> [#uses=2]
        store i32 %Y, i32* %P
        br label %Dead
}


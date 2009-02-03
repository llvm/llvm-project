; RUN: llvm-as < %s | opt -indvars | llvm-dis | grep indvar

@G = global i64 0               ; <i64*> [#uses=1]

define void @test(i64 %V) {
; <label>:0
        br label %Loop

Loop:           ; preds = %Loop, %0
        %X = phi i64 [ 1, %0 ], [ %X.next, %Loop ]              ; <i64> [#uses=2]
        %X.next = sub i64 %X, %V                ; <i64> [#uses=1]
        store i64 %X, i64* @G
        br label %Loop
}


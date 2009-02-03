; RUN: llvm-as < %s | opt -indvars | llvm-dis | grep indvar | not grep i32

@G = global i64 0               ; <i64*> [#uses=1]

define void @test() {
; <label>:0
        br label %Loop

Loop:           ; preds = %Loop, %0
        %X = phi i64 [ 1, %0 ], [ %X.next, %Loop ]              ; <i64> [#uses=2]
        %X.next = add i64 %X, 1         ; <i64> [#uses=1]
        store i64 %X, i64* @G
        br label %Loop
}


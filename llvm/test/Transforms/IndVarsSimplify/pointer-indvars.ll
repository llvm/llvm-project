; RUN: llvm-as < %s | opt -indvars | llvm-dis | grep indvar
@G = global i32* null           ; <i32**> [#uses=1]
@Array = external global [40 x i32]             ; <[40 x i32]*> [#uses=1]

define void @test() {
; <label>:0
        br label %Loop

Loop:           ; preds = %Loop, %0
        %X = phi i32* [ getelementptr ([40 x i32]* @Array, i64 0, i64 0), %0 ], [ %X.next, %Loop ]              ; <i32*> [#uses=2]
        %X.next = getelementptr i32* %X, i64 1          ; <i32*> [#uses=1]
        store i32* %X, i32** @G
        br label %Loop
}


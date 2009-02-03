; RUN: llvm-as < %s | llc -march=c

; Indirect function call test... found by Joel & Brian
;

@taskArray = external global i32*               ; <i32**> [#uses=1]

define void @test(i32 %X) {
        %Y = add i32 %X, -1             ; <i32> [#uses=1]
        %cast100 = sext i32 %Y to i64           ; <i64> [#uses=1]
        %gep100 = getelementptr i32** @taskArray, i64 %cast100          ; <i32**> [#uses=1]
        %fooPtr = load i32** %gep100            ; <i32*> [#uses=1]
        %cast101 = bitcast i32* %fooPtr to void (i32)*          ; <void (i32)*> [#uses=1]
        call void %cast101( i32 1000 )
        ret void
}


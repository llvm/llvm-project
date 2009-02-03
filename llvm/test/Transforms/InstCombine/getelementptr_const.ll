; Test folding of constantexpr geps into normal geps.
; RUN: llvm-as < %s | opt -instcombine -gvn -instcombine | \
; RUN:    llvm-dis | not grep getelementptr

@Array = external global [40 x i32]             ; <[40 x i32]*> [#uses=2]

define i32 @test(i64 %X) {
        %A = getelementptr i32* getelementptr ([40 x i32]* @Array, i64 0, i64 0), i64 %X  ; <i32*> [#uses=1]
        %B = getelementptr [40 x i32]* @Array, i64 0, i64 %X            ; <i32*> [#uses=1]
        %a = ptrtoint i32* %A to i32            ; <i32> [#uses=1]
        %b = ptrtoint i32* %B to i32            ; <i32> [#uses=1]
        %c = sub i32 %a, %b             ; <i32> [#uses=1]
        ret i32 %c
}


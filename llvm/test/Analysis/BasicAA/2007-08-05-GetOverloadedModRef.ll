; PR1600
; RUN: llvm-as < %s | opt -basicaa -gvn -instcombine | llvm-dis | \
; RUN:   grep {ret i32 0}
; END.

declare i16 @llvm.cttz.i16(i16)

define i32 @test(i32* %P, i16* %Q) {
        %A = load i16* %Q               ; <i16> [#uses=1]
        %x = load i32* %P               ; <i32> [#uses=1]
        %B = call i16 @llvm.cttz.i16( i16 %A )          ; <i16> [#uses=1]
        %y = load i32* %P               ; <i32> [#uses=1]
        store i16 %B, i16* %Q
        %z = sub i32 %x, %y             ; <i32> [#uses=1]
        ret i32 %z
}


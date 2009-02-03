; A store or load cannot alias a global if the accessed amount is larger then
; the global.

; RUN: llvm-as < %s | opt -basicaa -gvn -instcombine | llvm-dis | not grep load

@B = global i16 8               ; <i16*> [#uses=2]

define i16 @test(i32* %P) {
        %X = load i16* @B               ; <i16> [#uses=1]
        store i32 7, i32* %P
        %Y = load i16* @B               ; <i16> [#uses=1]
        %Z = sub i16 %Y, %X             ; <i16> [#uses=1]
        ret i16 %Z
}


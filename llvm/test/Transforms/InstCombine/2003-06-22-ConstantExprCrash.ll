; This is a bug in the VMcode library, not instcombine, it's just convenient 
; to expose it here.

; RUN: llvm-as < %s | opt -instcombine -disable-output

@A = global i32 1               ; <i32*> [#uses=1]
@B = global i32 2               ; <i32*> [#uses=1]

define i1 @test() {
        %C = icmp ult i32* getelementptr (i32* @A, i64 1), getelementptr (i32* @B, i64 2) ; <i1> [#uses=1]
        ret i1 %C
}


; RUN: llvm-as < %s | opt -instcombine -disable-output

%Ty = type opaque

define i32 @test(%Ty* %X) {
        %Y = bitcast %Ty* %X to i32*            ; <i32*> [#uses=1]
        %Z = load i32* %Y               ; <i32> [#uses=1]
        ret i32 %Z
}


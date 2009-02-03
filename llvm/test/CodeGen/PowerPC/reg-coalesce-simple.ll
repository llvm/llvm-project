; RUN: llvm-as < %s | llc -march=ppc32  | not grep or

%struct.foo = type { i32, i32, [0 x i8] }

define i32 @test(%struct.foo* %X) nounwind {
        %tmp1 = getelementptr %struct.foo* %X, i32 0, i32 2, i32 100            ; <i8*> [#uses=1]
        %tmp = load i8* %tmp1           ; <i8> [#uses=1]
        %tmp2 = zext i8 %tmp to i32             ; <i32> [#uses=1]
        ret i32 %tmp2
}



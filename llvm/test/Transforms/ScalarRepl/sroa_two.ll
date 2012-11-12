; RUN: opt < %s -scalarrepl | llvm-dis

define i32 @test(i32 %X) {
	%Arr = alloca [2 x i32]		; <[2 x i32]*> [#uses=3]
	%tmp.0 = getelementptr [2 x i32]* %Arr, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 1, i32* %tmp.0
	%tmp.1 = getelementptr [2 x i32]* %Arr, i32 0, i32 1		; <i32*> [#uses=1]
	store i32 2, i32* %tmp.1
	%tmp.3 = getelementptr [2 x i32]* %Arr, i32 0, i32 %X		; <i32*> [#uses=1]
	%tmp.4 = load i32* %tmp.3		; <i32> [#uses=1]
	ret i32 %tmp.4
}


; RUN: llvm-as < %s | opt -gvn -instcombine | llvm-dis | grep sub

; BasicAA was incorrectly concluding that P1 and P2 didn't conflict!

define i32 @test(i32 *%Ptr, i64 %V) {
	%P2 = getelementptr i32* %Ptr, i64 1
	%P1 = getelementptr i32* %Ptr, i64 %V
	%X = load i32* %P1
	store i32 5, i32* %P2

	%Y = load i32* %P1

	%Z = sub i32 %X, %Y
	ret i32 %Z
}

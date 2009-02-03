; RUN: llvm-as < %s | llc -march=arm -stats |& grep asm-printer | grep 4

define i32 @t1(i32 %a) {
	%b = mul i32 %a, 9
        %c = inttoptr i32 %b to i32*
        %d = load i32* %c
	ret i32 %d
}

define i32 @t2(i32 %a) {
	%b = mul i32 %a, -7
        %c = inttoptr i32 %b to i32*
        %d = load i32* %c
	ret i32 %d
}

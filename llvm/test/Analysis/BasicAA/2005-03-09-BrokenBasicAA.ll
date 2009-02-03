; RUN: llvm-as < %s | opt -basicaa -gvn -instcombine |\
; RUN: llvm-dis | grep {load i32\\* %A}

declare double* @useit(i32*)

define i32 @foo(i32 %Amt) {
	%A = malloc i32, i32 %Amt
	%P = call double*  @useit(i32* %A)

	%X = load i32* %A
	store double 0.0, double* %P
	%Y = load i32* %A
	%Z = sub i32 %X, %Y
	ret i32 %Z
}

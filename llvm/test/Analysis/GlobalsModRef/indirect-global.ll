; RUN: llvm-as < %s | opt -globalsmodref-aa -gvn -instcombine | llvm-dis | \
; RUN:   grep {ret i32 0}

@G = internal global i32* null		; <i32**> [#uses=3]

define void @test() {
	%A = malloc i32		; <i32*> [#uses=1]
	store i32* %A, i32** @G
	ret void
}

define i32 @test1(i32* %P) {
	%g1 = load i32** @G		; <i32*> [#uses=2]
	%h1 = load i32* %g1		; <i32> [#uses=1]
	store i32 123, i32* %P
	%g2 = load i32** @G		; <i32*> [#uses=0]
	%h2 = load i32* %g1		; <i32> [#uses=1]
	%X = sub i32 %h1, %h2		; <i32> [#uses=1]
	ret i32 %X
}

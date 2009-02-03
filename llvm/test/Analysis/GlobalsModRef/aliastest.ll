; RUN: llvm-as < %s | opt -globalsmodref-aa -gvn | llvm-dis | not grep load
@X = internal global i32 4		; <i32*> [#uses=1]

define i32 @test(i32* %P) {
	store i32 7, i32* %P
	store i32 12, i32* @X
	%V = load i32* %P		; <i32> [#uses=1]
	ret i32 %V
}

; RUN: llvm-as < %s | opt -globalsmodref-aa -gvn | llvm-dis | not grep load
@X = internal global i32 4		; <i32*> [#uses=2]

define i32 @test(i32* %P) {
	store i32 12, i32* @X
	call void @doesnotmodX( )
	%V = load i32* @X		; <i32> [#uses=1]
	ret i32 %V
}

define void @doesnotmodX() {
	ret void
}

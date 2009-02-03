; RUN: llvm-as < %s | llc -march=x86 | not grep lea
; RUN: llvm-as < %s | llc -march=x86 -mtriple=i686-apple-darwin8 | \
; RUN:   grep {movl	\$4, (.*,.*,4)}

define i32 @test(i32* %X, i32 %B) {
	; This gep should be sunk out of this block into the load/store users.
	%P = getelementptr i32* %X, i32 %B
	%G = icmp ult i32 %B, 1234
	br i1 %G, label %T, label %F
T:
	store i32 4, i32* %P
	ret i32 141
F:
	%V = load i32* %P
	ret i32 %V
}
	
	

; RUN: not llvm-as < %s |& grep {invalid getelementptr indices}

; This testcase is invalid because we are indexing into a pointer that is 
; contained WITHIN a structure.

define void @test({i32, i32*} * %X) {
	getelementptr {i32, i32*} * %X, i32 0, i32 1, i32 0
	ret void
}

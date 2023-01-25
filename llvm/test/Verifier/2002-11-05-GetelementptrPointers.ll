; RUN: not llvm-as < %s 2>&1 | FileCheck %s
; CHECK: invalid getelementptr indices

; This testcase is invalid because we are indexing into a pointer that is 
; contained WITHIN a structure.

define void @test(ptr %X) {
	getelementptr {i32, ptr}, ptr %X, i32 0, i32 1, i32 0
	ret void
}

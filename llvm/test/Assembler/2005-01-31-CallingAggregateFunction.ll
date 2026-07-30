; RUN: llvm-as %s -o /dev/null
; RUN: verify-uselistorder %s

define void @test() {
	call {i32} @foo()
	ret void
}

declare {i32 } @foo()

; RUN: llvm-as %s -disable-output
; RUN: verify-uselistorder %s

define void @test() {
	call {i32} @foo()
	ret void
}

declare {i32 } @foo()

; RUN: llvm-as %s -o /dev/null

define void @test() {
	call {i32} @foo()
	ret void
}

declare {i32 } @foo()

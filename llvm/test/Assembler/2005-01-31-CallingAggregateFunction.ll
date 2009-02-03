; RUN: llvm-as %s -o /dev/null -f 

define void @test() {
	call {i32} @foo()
	ret void
}

declare {i32 } @foo()

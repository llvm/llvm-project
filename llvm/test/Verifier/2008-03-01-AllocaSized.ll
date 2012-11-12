; RUN: not llvm-as %s -o /dev/null 2>&1 | grep "Cannot allocate unsized type"
; PR2113

define void @test() {
	%A = alloca void()
	ret void
}


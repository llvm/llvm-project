; RUN: llvm-as %s -o /dev/null -f

%T = type i32 *

define %T @test() {
	ret %T null
}

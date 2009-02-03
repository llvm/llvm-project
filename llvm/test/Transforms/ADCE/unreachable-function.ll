; RUN: llvm-as < %s | opt -adce -disable-output

define void @test() {
	unreachable
}

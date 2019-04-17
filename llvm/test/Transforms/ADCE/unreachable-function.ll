; RUN: opt < %s -adce -disable-output

define void @test() {
	unreachable
}

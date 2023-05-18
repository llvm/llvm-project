; RUN: opt < %s -passes=adce -disable-output

define void @test() {
	unreachable
}

; RUN: llc < %s -march=x86 -mattr=-sse2 | grep lock

define void @test() {
	fence seq_cst
	ret void
}

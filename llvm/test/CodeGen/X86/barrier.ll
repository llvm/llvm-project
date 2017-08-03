; RUN: llc < %s -march=x86 -mattr=-sse2 | FileCheck %s

define void @test() {
; CHECK: lock
	fence seq_cst
	ret void
}

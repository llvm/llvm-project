; RUN: llc < %s -mtriple=xcore > %t1.s
define void @store32(ptr %p) nounwind {
entry:
	store i192 0, ptr %p, align 4
	ret void
}

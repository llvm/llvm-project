; RUN: llc < %s -march=xcore > %t1.s
define void @store32(i8* %p) nounwind {
entry:
	%0 = bitcast i8* %p to i192*
	store i192 0, i192* %0, align 4
	ret void
}

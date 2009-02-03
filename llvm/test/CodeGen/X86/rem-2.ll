; RUN: llvm-as < %s | llc -march=x86 | not grep cltd

define i32 @test(i32 %X) nounwind readnone {
entry:
	%0 = srem i32 41, %X
	ret i32 %0
}

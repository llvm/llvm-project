; RUN: llvm-as < %s | llc -march=x86-64 | not grep movw

define i16 @test5(i16 %f12) nounwind {
	%f11 = shl i16 %f12, 2		; <i16> [#uses=1]
	%tmp7.25 = ashr i16 %f11, 8		; <i16> [#uses=1]
	ret i16 %tmp7.25
}

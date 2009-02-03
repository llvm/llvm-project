; RUN: llvm-as < %s | llc -march=x86 -mattr=+mmx,+sse2
; PR2850

@tmp_V2i = common global <2 x i32> zeroinitializer		; <<2 x i32>*> [#uses=2]

define void @f0() nounwind {
entry:
	%0 = load <2 x i32>* @tmp_V2i, align 8		; <<2 x i32>> [#uses=1]
	%1 = shufflevector <2 x i32> %0, <2 x i32> undef, <2 x i32> zeroinitializer		; <<2 x i32>> [#uses=1]
	store <2 x i32> %1, <2 x i32>* @tmp_V2i, align 8
	ret void
}

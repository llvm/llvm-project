; RUN: llvm-as < %s | llc -march=x86-64 -mattr=+sse2

define void @test() {
	%tmp1 = call <8 x i16> @llvm.x86.sse2.pmins.w( <8 x i16> zeroinitializer, <8 x i16> bitcast (<4 x i32> < i32 7, i32 7, i32 7, i32 7 > to <8 x i16>) )
	%tmp2 = bitcast <8 x i16> %tmp1 to <4 x i32>
	br i1 false, label %bb1, label %bb2

bb2:
	%tmp38007.i = extractelement <4 x i32> %tmp2, i32 3
	ret void

bb1:
	ret void
}

declare <8 x i16> @llvm.x86.sse2.pmins.w(<8 x i16>, <8 x i16>)

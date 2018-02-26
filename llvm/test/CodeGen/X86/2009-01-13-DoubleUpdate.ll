; RUN: llc < %s -mtriple=i686-- -mattr=+sse2 -enable-legalize-types-checking

declare <2 x double> @llvm.x86.sse2.min.pd(<2 x double>, <2 x double>) nounwind readnone

define void @__mindd16(<16 x double>* sret %vec.result, <16 x double> %x, double %y) nounwind {
entry:
	%tmp3.i = shufflevector <16 x double> zeroinitializer, <16 x double> undef, <8 x i32> < i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7 >		; <<8 x double>> [#uses=1]
	%tmp10.i.i = shufflevector <8 x double> %tmp3.i, <8 x double> undef, <4 x i32> < i32 4, i32 5, i32 6, i32 7 >		; <<4 x double>> [#uses=1]
	%tmp3.i2.i.i = shufflevector <4 x double> %tmp10.i.i, <4 x double> undef, <2 x i32> < i32 0, i32 1 >		; <<2 x double>> [#uses=1]
	%0 = tail call <2 x double> @llvm.x86.sse2.min.pd(<2 x double> zeroinitializer, <2 x double> %tmp3.i2.i.i) nounwind		; <<2 x double>> [#uses=1]
	%tmp5.i3.i.i = shufflevector <2 x double> %0, <2 x double> undef, <4 x i32> < i32 0, i32 1, i32 undef, i32 undef >		; <<4 x double>> [#uses=1]
	%tmp6.i4.i.i = shufflevector <4 x double> zeroinitializer, <4 x double> %tmp5.i3.i.i, <4 x i32> < i32 4, i32 5, i32 2, i32 3 >		; <<4 x double>> [#uses=1]
	%tmp14.i8.i.i = shufflevector <4 x double> %tmp6.i4.i.i, <4 x double> zeroinitializer, <4 x i32> < i32 0, i32 1, i32 4, i32 5 >		; <<4 x double>> [#uses=1]
	%tmp13.i.i = shufflevector <4 x double> %tmp14.i8.i.i, <4 x double> undef, <8 x i32> < i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef >		; <<8 x double>> [#uses=1]
	%tmp14.i.i = shufflevector <8 x double> zeroinitializer, <8 x double> %tmp13.i.i, <8 x i32> < i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11 >		; <<8 x double>> [#uses=1]
	%tmp5.i = shufflevector <8 x double> %tmp14.i.i, <8 x double> undef, <16 x i32> < i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef >		; <<16 x double>> [#uses=1]
	%tmp6.i = shufflevector <16 x double> %x, <16 x double> %tmp5.i, <16 x i32> < i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15 >		; <<16 x double>> [#uses=1]
	%tmp14.i = shufflevector <16 x double> %tmp6.i, <16 x double> zeroinitializer, <16 x i32> < i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23 >		; <<16 x double>> [#uses=1]
	store <16 x double> %tmp14.i, <16 x double>* %vec.result
	ret void
}

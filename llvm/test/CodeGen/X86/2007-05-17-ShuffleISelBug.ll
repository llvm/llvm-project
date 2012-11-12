; RUN: llc < %s -march=x86 -mattr=+sse2
; RUN: llc < %s -march=x86 -mattr=+sse2 | not grep punpckhwd

declare <16 x i8> @llvm.x86.sse2.packuswb.128(<8 x i16>, <8 x i16>)

declare <8 x i16> @llvm.x86.sse2.psrl.w(<8 x i16>, <8 x i16>)

define fastcc void @test(i32* %src, i32 %sbpr, i32* %dst, i32 %dbpr, i32 %w, i32 %h, i32 %dstalpha, i32 %mask) {
	%tmp633 = shufflevector <8 x i16> zeroinitializer, <8 x i16> undef, <8 x i32> < i32 4, i32 4, i32 5, i32 5, i32 6, i32 6, i32 7, i32 7 >
	%tmp715 = mul <8 x i16> zeroinitializer, %tmp633
	%tmp776 = bitcast <8 x i16> %tmp715 to <4 x i32>
	%tmp777 = add <4 x i32> %tmp776, shufflevector (<4 x i32> < i32 65537, i32 0, i32 0, i32 0 >, <4 x i32> < i32 65537, i32 0, i32 0, i32 0 >, <4 x i32> zeroinitializer)
	%tmp805 = add <4 x i32> %tmp777, zeroinitializer
	%tmp832 = bitcast <4 x i32> %tmp805 to <8 x i16>
	%tmp838 = tail call <8 x i16> @llvm.x86.sse2.psrl.w( <8 x i16> %tmp832, <8 x i16> < i16 8, i16 undef, i16 undef, i16 undef, i16 undef, i16 undef, i16 undef, i16 undef > )
	%tmp1020 = tail call <16 x i8> @llvm.x86.sse2.packuswb.128( <8 x i16> zeroinitializer, <8 x i16> %tmp838 )
	%tmp1030 = bitcast <16 x i8> %tmp1020 to <4 x i32>
	%tmp1033 = add <4 x i32> zeroinitializer, %tmp1030
	%tmp1048 = bitcast <4 x i32> %tmp1033 to <2 x i64>
	%tmp1049 = or <2 x i64> %tmp1048, zeroinitializer
	store <2 x i64> %tmp1049, <2 x i64>* null
	ret void
}

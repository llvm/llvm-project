; RUN: llvm-as < %s | llc -march=x86 -mattr=+mmx | grep punpckhdq | count 1

define void @bork(<1 x i64>* %x) {
entry:
	%tmp2 = load <1 x i64>* %x		; <<1 x i64>> [#uses=1]
	%tmp6 = bitcast <1 x i64> %tmp2 to <2 x i32>		; <<2 x i32>> [#uses=1]
	%tmp9 = shufflevector <2 x i32> %tmp6, <2 x i32> undef, <2 x i32> < i32 1, i32 1 >		; <<2 x i32>> [#uses=1]
	%tmp10 = bitcast <2 x i32> %tmp9 to <1 x i64>		; <<1 x i64>> [#uses=1]
	store <1 x i64> %tmp10, <1 x i64>* %x
	tail call void @llvm.x86.mmx.emms( )
	ret void
}

declare void @llvm.x86.mmx.emms()

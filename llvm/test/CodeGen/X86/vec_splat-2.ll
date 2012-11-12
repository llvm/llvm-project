; RUN: llc < %s -march=x86 -mcpu=penryn -mattr=+sse2 | grep pshufd | count 1

define void @test(<2 x i64>* %P, i8 %x) nounwind {
	%tmp = insertelement <16 x i8> zeroinitializer, i8 %x, i32 0		; <<16 x i8>> [#uses=1]
	%tmp36 = insertelement <16 x i8> %tmp, i8 %x, i32 1		; <<16 x i8>> [#uses=1]
	%tmp38 = insertelement <16 x i8> %tmp36, i8 %x, i32 2		; <<16 x i8>> [#uses=1]
	%tmp40 = insertelement <16 x i8> %tmp38, i8 %x, i32 3		; <<16 x i8>> [#uses=1]
	%tmp42 = insertelement <16 x i8> %tmp40, i8 %x, i32 4		; <<16 x i8>> [#uses=1]
	%tmp44 = insertelement <16 x i8> %tmp42, i8 %x, i32 5		; <<16 x i8>> [#uses=1]
	%tmp46 = insertelement <16 x i8> %tmp44, i8 %x, i32 6		; <<16 x i8>> [#uses=1]
	%tmp48 = insertelement <16 x i8> %tmp46, i8 %x, i32 7		; <<16 x i8>> [#uses=1]
	%tmp50 = insertelement <16 x i8> %tmp48, i8 %x, i32 8		; <<16 x i8>> [#uses=1]
	%tmp52 = insertelement <16 x i8> %tmp50, i8 %x, i32 9		; <<16 x i8>> [#uses=1]
	%tmp54 = insertelement <16 x i8> %tmp52, i8 %x, i32 10		; <<16 x i8>> [#uses=1]
	%tmp56 = insertelement <16 x i8> %tmp54, i8 %x, i32 11		; <<16 x i8>> [#uses=1]
	%tmp58 = insertelement <16 x i8> %tmp56, i8 %x, i32 12		; <<16 x i8>> [#uses=1]
	%tmp60 = insertelement <16 x i8> %tmp58, i8 %x, i32 13		; <<16 x i8>> [#uses=1]
	%tmp62 = insertelement <16 x i8> %tmp60, i8 %x, i32 14		; <<16 x i8>> [#uses=1]
	%tmp64 = insertelement <16 x i8> %tmp62, i8 %x, i32 15		; <<16 x i8>> [#uses=1]
	%tmp68 = load <2 x i64>* %P		; <<2 x i64>> [#uses=1]
	%tmp71 = bitcast <2 x i64> %tmp68 to <16 x i8>		; <<16 x i8>> [#uses=1]
	%tmp73 = add <16 x i8> %tmp71, %tmp64		; <<16 x i8>> [#uses=1]
	%tmp73.upgrd.1 = bitcast <16 x i8> %tmp73 to <2 x i64>		; <<2 x i64>> [#uses=1]
	store <2 x i64> %tmp73.upgrd.1, <2 x i64>* %P
	ret void
}

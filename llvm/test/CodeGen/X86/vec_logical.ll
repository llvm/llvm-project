; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 > %t
; RUN: grep xorps %t | count 2
; RUN: grep andnps %t
; RUN: grep movaps %t | count 2

define void @t(<4 x float> %A) {
	%tmp1277 = sub <4 x float> < float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00 >, %A
	store <4 x float> %tmp1277, <4 x float>* null
	ret void
}

define <4 x float> @t1(<4 x float> %a, <4 x float> %b) {
entry:
	%tmp9 = bitcast <4 x float> %a to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp10 = bitcast <4 x float> %b to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp11 = xor <4 x i32> %tmp9, %tmp10		; <<4 x i32>> [#uses=1]
	%tmp13 = bitcast <4 x i32> %tmp11 to <4 x float>		; <<4 x float>> [#uses=1]
	ret <4 x float> %tmp13
}

define <2 x double> @t2(<2 x double> %a, <2 x double> %b) {
entry:
	%tmp9 = bitcast <2 x double> %a to <2 x i64>		; <<2 x i64>> [#uses=1]
	%tmp10 = bitcast <2 x double> %b to <2 x i64>		; <<2 x i64>> [#uses=1]
	%tmp11 = and <2 x i64> %tmp9, %tmp10		; <<2 x i64>> [#uses=1]
	%tmp13 = bitcast <2 x i64> %tmp11 to <2 x double>		; <<2 x double>> [#uses=1]
	ret <2 x double> %tmp13
}

define void @t3(<4 x float> %a, <4 x float> %b, <4 x float>* %c, <4 x float>* %d) {
entry:
	%tmp3 = load <4 x float>* %c		; <<4 x float>> [#uses=1]
	%tmp11 = bitcast <4 x float> %a to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp12 = bitcast <4 x float> %b to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp13 = xor <4 x i32> %tmp11, < i32 -1, i32 -1, i32 -1, i32 -1 >		; <<4 x i32>> [#uses=1]
	%tmp14 = and <4 x i32> %tmp12, %tmp13		; <<4 x i32>> [#uses=1]
	%tmp27 = bitcast <4 x float> %tmp3 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp28 = or <4 x i32> %tmp14, %tmp27		; <<4 x i32>> [#uses=1]
	%tmp30 = bitcast <4 x i32> %tmp28 to <4 x float>		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp30, <4 x float>* %d
	ret void
}

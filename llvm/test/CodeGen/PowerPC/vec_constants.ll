; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 | not grep CPI

define void @test1(<4 x i32>* %P1, <4 x i32>* %P2, <4 x float>* %P3) {
	%tmp = load <4 x i32>* %P1		; <<4 x i32>> [#uses=1]
	%tmp4 = and <4 x i32> %tmp, < i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648 >		; <<4 x i32>> [#uses=1]
	store <4 x i32> %tmp4, <4 x i32>* %P1
	%tmp7 = load <4 x i32>* %P2		; <<4 x i32>> [#uses=1]
	%tmp9 = and <4 x i32> %tmp7, < i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647 >		; <<4 x i32>> [#uses=1]
	store <4 x i32> %tmp9, <4 x i32>* %P2
	%tmp.upgrd.1 = load <4 x float>* %P3		; <<4 x float>> [#uses=1]
	%tmp11 = bitcast <4 x float> %tmp.upgrd.1 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp12 = and <4 x i32> %tmp11, < i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647 >		; <<4 x i32>> [#uses=1]
	%tmp13 = bitcast <4 x i32> %tmp12 to <4 x float>		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp13, <4 x float>* %P3
	ret void
}

define <4 x i32> @test_30() {
	ret <4 x i32> < i32 30, i32 30, i32 30, i32 30 >
}

define <4 x i32> @test_29() {
	ret <4 x i32> < i32 29, i32 29, i32 29, i32 29 >
}

define <8 x i16> @test_n30() {
	ret <8 x i16> < i16 -30, i16 -30, i16 -30, i16 -30, i16 -30, i16 -30, i16 -30, i16 -30 >
}

define <16 x i8> @test_n104() {
	ret <16 x i8> < i8 -104, i8 -104, i8 -104, i8 -104, i8 -104, i8 -104, i8 -104, i8 -104, i8 -104, i8 -104, i8 -104, i8 -104, i8 -104, i8 -104, i8 -104, i8 -104 >
}

define <4 x i32> @test_vsldoi() {
	ret <4 x i32> < i32 512, i32 512, i32 512, i32 512 >
}

define <4 x i32> @test_rol() {
	ret <4 x i32> < i32 -11534337, i32 -11534337, i32 -11534337, i32 -11534337 >
}

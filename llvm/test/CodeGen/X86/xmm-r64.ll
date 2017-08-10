; RUN: llc < %s -mtriple=x86_64--

define <4 x i32> @test() {
        %tmp1039 = call <4 x i32> @llvm.x86.sse2.psll.d( <4 x i32> zeroinitializer, <4 x i32> zeroinitializer )               ; <<4 x i32>> [#uses=1]
        %tmp1040 = bitcast <4 x i32> %tmp1039 to <2 x i64>              ; <<2 x i64>> [#uses=1]
        %tmp1048 = add <2 x i64> %tmp1040, zeroinitializer              ; <<2 x i64>> [#uses=1]
        %tmp1048.upgrd.1 = bitcast <2 x i64> %tmp1048 to <4 x i32>              ; <<4 x i32>> [#uses=1]
        ret <4 x i32> %tmp1048.upgrd.1
}

declare <4 x i32> @llvm.x86.sse2.psll.d(<4 x i32>, <4 x i32>)


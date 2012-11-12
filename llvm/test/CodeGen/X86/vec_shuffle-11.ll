; RUN: llc < %s -march=x86 -mattr=+sse2 
; RUN: llc < %s -march=x86 -mattr=+sse2 -mtriple=i386-apple-darwin | not grep mov

define <4 x i32> @test() nounwind {
        %tmp131 = call <2 x i64> @llvm.x86.sse2.psrl.dq( <2 x i64> < i64 -1, i64 -1 >, i32 96 )         ; <<2 x i64>> [#uses=1]
        %tmp137 = bitcast <2 x i64> %tmp131 to <4 x i32>                ; <<4 x i32>> [#uses=1]
        %tmp138 = and <4 x i32> %tmp137, bitcast (<2 x i64> < i64 -1, i64 -1 > to <4 x i32>)            ; <<4 x i32>> [#uses=1]
        ret <4 x i32> %tmp138
}

declare <2 x i64> @llvm.x86.sse2.psrl.dq(<2 x i64>, i32)

; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 -o %t  -f
; RUN: grep xorps %t | count 1
; RUN: not grep shufps %t

define void @test() {
        bitcast <4 x i32> zeroinitializer to <4 x float>                ; <<4 x float>>:1 [#uses=1]
        shufflevector <4 x float> %1, <4 x float> zeroinitializer, <4 x i32> zeroinitializer         ; <<4 x float>>:2 [#uses=1]
        store <4 x float> %2, <4 x float>* null
        unreachable
}


; RUN: llc < %s -march=x86-64 -enable-unsafe-fp-math | not grep mulps
; RUN: llc < %s -march=x86-64 | grep mulps

define void @test14(<4 x float>*) nounwind {
        load <4 x float>* %0, align 1
        fmul <4 x float> %2, zeroinitializer
        store <4 x float> %3, <4 x float>* %0, align 1
        ret void
}

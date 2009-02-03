; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse

define void @test(i32 %C, <4 x float>* %A, <4 x float>* %B) {
        %tmp = load <4 x float>* %A             ; <<4 x float>> [#uses=1]
        %tmp3 = load <4 x float>* %B            ; <<4 x float>> [#uses=2]
        %tmp9 = mul <4 x float> %tmp3, %tmp3            ; <<4 x float>> [#uses=1]
        %tmp.upgrd.1 = icmp eq i32 %C, 0                ; <i1> [#uses=1]
        %iftmp.38.0 = select i1 %tmp.upgrd.1, <4 x float> %tmp9, <4 x float> %tmp               ; <<4 x float>> [#uses=1]
        store <4 x float> %iftmp.38.0, <4 x float>* %A
        ret void
}


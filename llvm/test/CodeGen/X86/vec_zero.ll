; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep xorps | count 2

define void @foo(<4 x float>* %P) {
        %T = load <4 x float>* %P               ; <<4 x float>> [#uses=1]
        %S = add <4 x float> zeroinitializer, %T                ; <<4 x float>> [#uses=1]
        store <4 x float> %S, <4 x float>* %P
        ret void
}

define void @bar(<4 x i32>* %P) {
        %T = load <4 x i32>* %P         ; <<4 x i32>> [#uses=1]
        %S = add <4 x i32> zeroinitializer, %T          ; <<4 x i32>> [#uses=1]
        store <4 x i32> %S, <4 x i32>* %P
        ret void
}


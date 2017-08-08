; RUN: llc -verify-machineinstrs < %s -march=ppc32 -mcpu=g5 | grep vxor

define void @foo(<4 x float>* %P) {
        %T = load <4 x float>, <4 x float>* %P               ; <<4 x float>> [#uses=1]
        %S = fadd <4 x float> zeroinitializer, %T                ; <<4 x float>> [#uses=1]
        store <4 x float> %S, <4 x float>* %P
        ret void
}


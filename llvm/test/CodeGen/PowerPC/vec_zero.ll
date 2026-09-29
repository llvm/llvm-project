; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- -mcpu=g5 | grep vxor

define void @foo(ptr %P) {
        %T = load <4 x float>, ptr %P               ; <<4 x float>> [#uses=1]
        %S = fadd <4 x float> zeroinitializer, %T                ; <<4 x float>> [#uses=1]
        store <4 x float> %S, ptr %P
        ret void
}


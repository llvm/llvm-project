; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- -mcpu=g5

define void @glgRunProcessor15() {
        %tmp26355.i = shufflevector <4 x float> zeroinitializer, <4 x float> < float 0x379FFFE000000000, float 0x379FFFE000000000, float 0x379FFFE000000000, float 0x379FFFE000000000 >, <4 x i32> < i32 0, i32 1, i32 2, i32 7 >; <<4 x float>> [#uses=1]
        %tmp3030030304.i = bitcast <4 x float> %tmp26355.i to <8 x i16>         ; <<8 x i16>> [#uses=1]
        %tmp30305.i = shufflevector <8 x i16> zeroinitializer, <8 x i16> %tmp3030030304.i, <8 x i32> < i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15 >               ; <<8 x i16>> [#uses=1]
        %tmp30305.i.upgrd.1 = bitcast <8 x i16> %tmp30305.i to <4 x i32>                ; <<4 x i32>> [#uses=1]
        store <4 x i32> %tmp30305.i.upgrd.1, ptr null
        ret void
}

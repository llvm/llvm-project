; There should be exactly one vxor here.
; RUN: llc < %s -march=ppc32 -mcpu=g5 --enable-unsafe-fp-math | \
; RUN:   grep vxor | count 1

; There should be exactly one vsplti here.
; RUN: llc < %s -march=ppc32 -mcpu=g5 --enable-unsafe-fp-math | \
; RUN:   grep vsplti | count 1

define void @VXOR(<4 x float>* %P1, <4 x i32>* %P2, <4 x float>* %P3) {
        %tmp = load <4 x float>* %P3            ; <<4 x float>> [#uses=1]
        %tmp3 = load <4 x float>* %P1           ; <<4 x float>> [#uses=1]
        %tmp4 = fmul <4 x float> %tmp, %tmp3             ; <<4 x float>> [#uses=1]
        store <4 x float> %tmp4, <4 x float>* %P3
        store <4 x float> zeroinitializer, <4 x float>* %P1
        store <4 x i32> zeroinitializer, <4 x i32>* %P2
        ret void
}

define void @VSPLTI(<4 x i32>* %P2, <8 x i16>* %P3) {
        store <4 x i32> bitcast (<16 x i8> < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1 > to <4 x i32>), <4 x i32>* %P2
        store <8 x i16> < i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1 >, <8 x i16>* %P3
        ret void
}


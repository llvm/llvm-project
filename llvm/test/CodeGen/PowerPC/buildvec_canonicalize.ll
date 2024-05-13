; RUN: llc -verify-machineinstrs < %s -mattr=-vsx -mtriple=ppc32-- -mattr=+altivec | FileCheck %s

define void @VXOR(ptr %P1, ptr %P2, ptr %P3) {
        %tmp = load <4 x float>, ptr %P3            ; <<4 x float>> [#uses=1]
        %tmp3 = load <4 x float>, ptr %P1           ; <<4 x float>> [#uses=1]
        %tmp4 = fmul <4 x float> %tmp, %tmp3             ; <<4 x float>> [#uses=1]
        store <4 x float> %tmp4, ptr %P3
        store <4 x float> zeroinitializer, ptr %P1
        store <4 x i32> zeroinitializer, ptr %P2
        ret void
}
; The fmul will spill a vspltisw to create a -0.0 vector used as the addend
; to vmaddfp (so it would IEEE compliant with zero sign propagation).
; CHECK: @VXOR
; CHECK: vsplti
; CHECK: vxor

define void @VSPLTI(ptr %P2, ptr %P3) {
        store <4 x i32> bitcast (<16 x i8> < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1 > to <4 x i32>), ptr %P2
        store <8 x i16> < i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1 >, ptr %P3
        ret void
}
; CHECK: @VSPLTI
; CHECK: vsplti

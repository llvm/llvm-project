; RUN: llc -verify-machineinstrs < %s -march=ppc32 -mcpu=g5 | grep li.*16
; RUN: llc -verify-machineinstrs < %s -march=ppc32 -mcpu=g5 | not grep addi

; Codegen lvx (R+16) as t = li 16,  lvx t,R
; This shares the 16 between the two loads.

define void @func(<4 x float>* %a, <4 x float>* %b) {
        %tmp1 = getelementptr <4 x float>, <4 x float>* %b, i32 1            ; <<4 x float>*> [#uses=1]
        %tmp = load <4 x float>, <4 x float>* %tmp1          ; <<4 x float>> [#uses=1]
        %tmp3 = getelementptr <4 x float>, <4 x float>* %a, i32 1            ; <<4 x float>*> [#uses=1]
        %tmp4 = load <4 x float>, <4 x float>* %tmp3         ; <<4 x float>> [#uses=1]
        %tmp5 = fmul <4 x float> %tmp, %tmp4             ; <<4 x float>> [#uses=1]
        %tmp8 = load <4 x float>, <4 x float>* %b            ; <<4 x float>> [#uses=1]
        %tmp9 = fadd <4 x float> %tmp5, %tmp8            ; <<4 x float>> [#uses=1]
        store <4 x float> %tmp9, <4 x float>* %a
        ret void
}


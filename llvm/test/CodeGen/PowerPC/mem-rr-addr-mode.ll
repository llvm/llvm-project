; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- -mcpu=g5 | grep li.*16
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- -mcpu=g5 | not grep addi

; Codegen lvx (R+16) as t = li 16,  lvx t,R
; This shares the 16 between the two loads.

define void @func(ptr %a, ptr %b) {
        %tmp1 = getelementptr <4 x float>, ptr %b, i32 1            ; <ptr> [#uses=1]
        %tmp = load <4 x float>, ptr %tmp1          ; <<4 x float>> [#uses=1]
        %tmp3 = getelementptr <4 x float>, ptr %a, i32 1            ; <ptr> [#uses=1]
        %tmp4 = load <4 x float>, ptr %tmp3         ; <<4 x float>> [#uses=1]
        %tmp5 = fmul <4 x float> %tmp, %tmp4             ; <<4 x float>> [#uses=1]
        %tmp8 = load <4 x float>, ptr %b            ; <<4 x float>> [#uses=1]
        %tmp9 = fadd <4 x float> %tmp5, %tmp8            ; <<4 x float>> [#uses=1]
        store <4 x float> %tmp9, ptr %a
        ret void
}


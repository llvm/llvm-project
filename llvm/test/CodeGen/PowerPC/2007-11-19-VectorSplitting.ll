; RUN: llc -verify-machineinstrs < %s
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- -mcpu=g3
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- -mcpu=g5
; PR1811
; REQUIRES: default_triple

define void @execute_shader(ptr %OUT, ptr %IN, ptr
%CONST) {
entry:
        %input2 = load <4 x float>, ptr null, align 16               ; <<4 x float>>
       	%shuffle7 = shufflevector <4 x float> %input2, <4 x float> < float 0.000000e+00, float 1.000000e+00, float 0.000000e+00, float 1.000000e+00 >, <4 x i32> < i32 2, i32 2, i32 2, i32 2 >		; <<4 x float>> [#uses=1]

        %mul1 = fmul <4 x float> %shuffle7, zeroinitializer              ; <<4 x
        %add2 = fadd <4 x float> %mul1, %input2          ; <<4 x float>>
        store <4 x float> %add2, ptr null, align 16
        ret void
}

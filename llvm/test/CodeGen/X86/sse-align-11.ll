; RUN: llc < %s -mcpu=yonah -mtriple=i686-apple-darwin8 | grep movaps
; RUN: llc < %s -mcpu=yonah -mtriple=i686-linux-gnu | grep movaps
; PR8969 - make 32-bit linux have a 16-byte aligned stack

define <4 x float> @foo(float %a, float %b, float %c, float %d) nounwind {
entry:
        %tmp6 = insertelement <4 x float> undef, float %a, i32 0               
        %tmp7 = insertelement <4 x float> %tmp6, float %b, i32 1               
        %tmp8 = insertelement <4 x float> %tmp7, float %c, i32 2               
        %tmp9 = insertelement <4 x float> %tmp8, float %d, i32 3               
        ret <4 x float> %tmp9
}


; RUN: llvm-as < %s | llc -march=x86 -mcpu=yonah -mtriple=i686-apple-darwin8 | grep movaps
; RUN: llvm-as < %s | llc -march=x86 -mcpu=yonah -mtriple=linux | grep movups

define <4 x float> @foo(float %a, float %b, float %c, float %d) nounwind {
entry:
        %tmp6 = insertelement <4 x float> undef, float %a, i32 0               
        %tmp7 = insertelement <4 x float> %tmp6, float %b, i32 1               
        %tmp8 = insertelement <4 x float> %tmp7, float %c, i32 2               
        %tmp9 = insertelement <4 x float> %tmp8, float %d, i32 3               
        ret <4 x float> %tmp9
}


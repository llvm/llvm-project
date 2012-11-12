; RUN: llc < %s -march=x86 -mattr=+sse2 | grep movss
; PR2472

define <4 x i32> @a(<4 x i32> %a) nounwind {
entry:
        %vecext = extractelement <4 x i32> %a, i32 0
        insertelement <4 x i32> zeroinitializer, i32 %vecext, i32 0
        %add = add <4 x i32> %a, %0
        ret <4 x i32> %add
}

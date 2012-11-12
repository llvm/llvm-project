; RUN: llc < %s -march=x86 -mcpu=yonah | grep "(%esp,%eax,4)" | count 4

; Inserts and extracts with variable indices must be lowered
; to memory accesses.

define i32 @t0(i32 inreg %t7, <4 x i32> inreg %t8) nounwind {
  %t13 = insertelement <4 x i32> %t8, i32 76, i32 %t7
  %t9 = extractelement <4 x i32> %t13, i32 0
  ret i32 %t9
}
define i32 @t1(i32 inreg %t7, <4 x i32> inreg %t8) nounwind {
  %t13 = insertelement <4 x i32> %t8, i32 76, i32 0
  %t9 = extractelement <4 x i32> %t13, i32 %t7
  ret i32 %t9
}
define <4 x i32> @t2(i32 inreg %t7, <4 x i32> inreg %t8) nounwind {
  %t9 = extractelement <4 x i32> %t8, i32 %t7
  %t13 = insertelement <4 x i32> %t8, i32 %t9, i32 0
  ret <4 x i32> %t13
}
define <4 x i32> @t3(i32 inreg %t7, <4 x i32> inreg %t8) nounwind {
  %t9 = extractelement <4 x i32> %t8, i32 0
  %t13 = insertelement <4 x i32> %t8, i32 %t9, i32 %t7
  ret <4 x i32> %t13
}

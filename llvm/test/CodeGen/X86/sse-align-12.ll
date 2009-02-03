; RUN: llvm-as < %s | llc -march=x86-64 > %t
; RUN: grep unpck %t | count 2
; RUN: grep shuf %t | count 2
; RUN: grep ps %t | count 4
; RUN: grep pd %t | count 4
; RUN: grep movup %t | count 4

define <4 x float> @a(<4 x float>* %y) nounwind {
  %x = load <4 x float>* %y, align 4
  %a = extractelement <4 x float> %x, i32 0
  %b = extractelement <4 x float> %x, i32 1
  %c = extractelement <4 x float> %x, i32 2
  %d = extractelement <4 x float> %x, i32 3
  %p = insertelement <4 x float> undef, float %d, i32 0
  %q = insertelement <4 x float> %p, float %c, i32 1
  %r = insertelement <4 x float> %q, float %b, i32 2
  %s = insertelement <4 x float> %r, float %a, i32 3
  ret <4 x float> %s
}
define <4 x float> @b(<4 x float>* %y, <4 x float> %z) nounwind {
  %x = load <4 x float>* %y, align 4
  %a = extractelement <4 x float> %x, i32 2
  %b = extractelement <4 x float> %x, i32 3
  %c = extractelement <4 x float> %z, i32 2
  %d = extractelement <4 x float> %z, i32 3
  %p = insertelement <4 x float> undef, float %c, i32 0
  %q = insertelement <4 x float> %p, float %a, i32 1
  %r = insertelement <4 x float> %q, float %d, i32 2
  %s = insertelement <4 x float> %r, float %b, i32 3
  ret <4 x float> %s
}
define <2 x double> @c(<2 x double>* %y) nounwind {
  %x = load <2 x double>* %y, align 8
  %a = extractelement <2 x double> %x, i32 0
  %c = extractelement <2 x double> %x, i32 1
  %p = insertelement <2 x double> undef, double %c, i32 0
  %r = insertelement <2 x double> %p, double %a, i32 1
  ret <2 x double> %r
}
define <2 x double> @d(<2 x double>* %y, <2 x double> %z) nounwind {
  %x = load <2 x double>* %y, align 8
  %a = extractelement <2 x double> %x, i32 1
  %c = extractelement <2 x double> %z, i32 1
  %p = insertelement <2 x double> undef, double %c, i32 0
  %r = insertelement <2 x double> %p, double %a, i32 1
  ret <2 x double> %r
}

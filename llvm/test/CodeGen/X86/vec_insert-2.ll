; RUN: llc < %s -march=x86 -mattr=+sse2,-sse41 | FileCheck --check-prefix=X32 %s
; RUN: llc < %s -march=x86-64 -mattr=+sse2,-sse41 | FileCheck --check-prefix=X64 %s

define <4 x float> @t1(float %s, <4 x float> %tmp) nounwind {
; X32: t1:
; X32: shufps $36
; X32: ret

  %tmp1 = insertelement <4 x float> %tmp, float %s, i32 3
  ret <4 x float> %tmp1
}

define <4 x i32> @t2(i32 %s, <4 x i32> %tmp) nounwind {
; X32: t2:
; X32: shufps $36
; X32: ret

  %tmp1 = insertelement <4 x i32> %tmp, i32 %s, i32 3
  ret <4 x i32> %tmp1
}

define <2 x double> @t3(double %s, <2 x double> %tmp) nounwind {
; X32: t3:
; X32: movhpd
; X32: ret

; X64: t3:
; X64: unpcklpd
; X64: ret

  %tmp1 = insertelement <2 x double> %tmp, double %s, i32 1
  ret <2 x double> %tmp1
}

define <8 x i16> @t4(i16 %s, <8 x i16> %tmp) nounwind {
; X32: t4:
; X32: pinsrw
; X32: ret

  %tmp1 = insertelement <8 x i16> %tmp, i16 %s, i32 5
  ret <8 x i16> %tmp1
}

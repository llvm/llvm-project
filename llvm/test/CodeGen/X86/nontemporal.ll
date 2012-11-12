; RUN: llc < %s -march=x86 -mattr=+sse2 | FileCheck %s

define void @f(<4 x float> %A, i8* %B, <2 x double> %C, i32 %D, <2 x i64> %E) {
; CHECK: movntps
  %cast = bitcast i8* %B to <4 x float>*
  %A2 = fadd <4 x float> %A, <float 0x0, float 0x0, float 0x0, float 0x4200000000000000>
  store <4 x float> %A2, <4 x float>* %cast, align 16, !nontemporal !0
; CHECK: movntdq
  %cast1 = bitcast i8* %B to <2 x i64>*
  %E2 = add <2 x i64> %E, <i64 1, i64 2>
  store <2 x i64> %E2, <2 x i64>* %cast1, align 16, !nontemporal !0
; CHECK: movntpd
  %cast2 = bitcast i8* %B to <2 x double>*
  %C2 = fadd <2 x double> %C, <double 0x0, double 0x4200000000000000>
  store <2 x double> %C2, <2 x double>* %cast2, align 16, !nontemporal !0
; CHECK: movnti
  %cast3 = bitcast i8* %B to i32*
  store i32 %D, i32* %cast3, align 16, !nontemporal !0
  ret void
}

!0 = metadata !{i32 1}

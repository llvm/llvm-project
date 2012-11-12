; RUN: llc < %s -march=x86 -mattr=+avx2 | FileCheck %s

define void @f(<8 x float> %A, i8* %B, <4 x double> %C, i32 %D, <4 x i64> %E) {
; CHECK: vmovntps
  %cast = bitcast i8* %B to <8 x float>*
  %A2 = fadd <8 x float> %A, <float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x0, float 0x4200000000000000>
  store <8 x float> %A2, <8 x float>* %cast, align 16, !nontemporal !0
; CHECK: vmovntdq
  %cast1 = bitcast i8* %B to <4 x i64>*
  %E2 = add <4 x i64> %E, <i64 1, i64 2, i64 3, i64 4>
  store <4 x i64> %E2, <4 x i64>* %cast1, align 16, !nontemporal !0
; CHECK: vmovntpd
  %cast2 = bitcast i8* %B to <4 x double>*
  %C2 = fadd <4 x double> %C, <double 0x0, double 0x0, double 0x0, double 0x4200000000000000>
  store <4 x double> %C2, <4 x double>* %cast2, align 16, !nontemporal !0
; CHECK: movnti
  %cast3 = bitcast i8* %B to i32*
  store i32 %D, i32* %cast3, align 16, !nontemporal !0
  ret void
}

!0 = metadata !{i32 1}

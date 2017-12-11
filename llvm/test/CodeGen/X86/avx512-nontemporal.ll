; RUN: llc < %s -mtriple=x86_64-- -mattr=+avx512f,+avx512bw | FileCheck %s

define i32 @f(<16 x float> %A, <16 x float> %AA, i8* %B, <8 x double> %C, <8 x double> %CC, <8 x i64> %E, <8 x i64> %EE, <16 x i32> %F, <16 x i32> %FF, <32 x i16> %G, <32 x i16> %GG, <64 x i8> %H, <64 x i8> %HH, i32 * %loadptr) {
; CHECK: vmovntps %z
  %v0 = load i32, i32* %loadptr, align 1
  %cast = bitcast i8* %B to <16 x float>*
  %A2 = fadd <16 x float> %A, %AA
  store <16 x float> %A2, <16 x float>* %cast, align 64, !nontemporal !0
  %v1 = load i32, i32* %loadptr, align 1
; CHECK: vmovntdq %z
  %cast1 = bitcast i8* %B to <8 x i64>*
  %E2 = add <8 x i64> %E, %EE
  store <8 x i64> %E2, <8 x i64>* %cast1, align 64, !nontemporal !0
  %v2 = load i32, i32* %loadptr, align 1
; CHECK: vmovntpd %z
  %cast2 = bitcast i8* %B to <8 x double>*
  %C2 = fadd <8 x double> %C, %CC
  store <8 x double> %C2, <8 x double>* %cast2, align 64, !nontemporal !0
  %v3 = load i32, i32* %loadptr, align 1
; CHECK: vmovntdq %z
  %cast3 = bitcast i8* %B to <16 x i32>*
  %F2 = add <16 x i32> %F, %FF
  store <16 x i32> %F2, <16 x i32>* %cast3, align 64, !nontemporal !0
  %v4 = load i32, i32* %loadptr, align 1
; CHECK: vmovntdq %z
  %cast4 = bitcast i8* %B to <32 x i16>*
  %G2 = add <32 x i16> %G, %GG
  store <32 x i16> %G2, <32 x i16>* %cast4, align 64, !nontemporal !0
  %v5 = load i32, i32* %loadptr, align 1
; CHECK: vmovntdq %z
  %cast5 = bitcast i8* %B to <64 x i8>*
  %H2 = add <64 x i8> %H, %HH
  store <64 x i8> %H2, <64 x i8>* %cast5, align 64, !nontemporal !0
  %v6 = load i32, i32* %loadptr, align 1
  %sum1 = add i32 %v0, %v1
  %sum2 = add i32 %sum1, %v2
  %sum3 = add i32 %sum2, %v3
  %sum4 = add i32 %sum3, %v4
  %sum5 = add i32 %sum4, %v5
  %sum6 = add i32 %sum5, %v6
  ret i32 %sum6
}

!0 = !{i32 1}

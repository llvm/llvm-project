; RUN: llc < %s -march=x86 -mattr=+avx | FileCheck %s


define <2 x double> @fabs_v2f64(<2 x double> %p)
{
  ; CHECK-LABEL: fabs_v2f64
  ; CHECK: vandps
  %t = call <2 x double> @llvm.fabs.v2f64(<2 x double> %p)
  ret <2 x double> %t
}
declare <2 x double> @llvm.fabs.v2f64(<2 x double> %p)

define <4 x float> @fabs_v4f32(<4 x float> %p)
{
  ; CHECK-LABEL: fabs_v4f32
  ; CHECK: vandps
  %t = call <4 x float> @llvm.fabs.v4f32(<4 x float> %p)
  ret <4 x float> %t
}
declare <4 x float> @llvm.fabs.v4f32(<4 x float> %p)

define <4 x double> @fabs_v4f64(<4 x double> %p)
{
  ; CHECK-LABEL: fabs_v4f64
  ; CHECK: vandps
  %t = call <4 x double> @llvm.fabs.v4f64(<4 x double> %p)
  ret <4 x double> %t
}
declare <4 x double> @llvm.fabs.v4f64(<4 x double> %p)

define <8 x float> @fabs_v8f32(<8 x float> %p)
{
  ; CHECK-LABEL: fabs_v8f32
  ; CHECK: vandps
  %t = call <8 x float> @llvm.fabs.v8f32(<8 x float> %p)
  ret <8 x float> %t
}
declare <8 x float> @llvm.fabs.v8f32(<8 x float> %p)

; PR20354: when generating code for a vector fabs op,
; make sure the correct mask is used for all vector elements.
; CHECK-LABEL: LCPI4_0
; CHECK: .long	2147483648
; CHECK: .long	2147483648
; CHECK-LABEL: LCPI4_1 
; CHECK: .long	2147483647
; CHECK: .long	2147483647
; CHECK-LABEL: fabs_v2f32_1
; CHECK: vmovdqa {{.*}}LCPI4_0, %xmm0
; CHECK: vpand   {{.*}}LCPI4_1, %xmm0, %xmm0
; CHECK: vmovd   %xmm0, %eax
; CHECK: vpextrd $1, %xmm0, %edx
define i64 @fabs_v2f32_1() {
  %highbits = bitcast i64 9223372039002259456 to <2 x float> ; 0x8000_0000_8000_0000
  %fabs = call <2 x float> @llvm.fabs.v2f32(<2 x float> %highbits)
  %ret = bitcast <2 x float> %fabs to i64
  ret i64 %ret
}

declare <2 x float> @llvm.fabs.v2f32(<2 x float> %p)

; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx | FileCheck %s

; This matter of this test is ensuring that vpackus* is not used for umin+trunc combination, since vpackus* input is a signed number.
define <16 x i8> @usat_trunc_wb_256(<16 x i16> %i) {
; CHECK-LABEL: usat_trunc_wb_256:
; CHECK-NOT:    vpackuswb %xmm1, %xmm0, %xmm0
  %x3 = icmp ult <16 x i16> %i, <i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255>
  %x5 = select <16 x i1> %x3, <16 x i16> %i, <16 x i16> <i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255>
  %x6 = trunc <16 x i16> %x5 to <16 x i8>
  ret <16 x i8> %x6
}
 
define <8 x i16> @usat_trunc_dw_256(<8 x i32> %i) {
; CHECK-LABEL: usat_trunc_dw_256:
; CHECK-NOT:    vpackusdw %xmm1, %xmm0, %xmm0
  %x3 = icmp ult <8 x i32> %i, <i32 65535, i32 65535, i32 65535, i32 65535, i32 65535, i32 65535, i32 65535, i32 65535>
  %x5 = select <8 x i1> %x3, <8 x i32> %i, <8 x i32> <i32 65535, i32 65535, i32 65535, i32 65535, i32 65535, i32 65535, i32 65535, i32 65535>
  %x6 = trunc <8 x i32> %x5 to <8 x i16>
  ret <8 x i16> %x6
}

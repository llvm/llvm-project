; RUN: llc -verify-machineinstrs -mtriple=powerpc64-linux-gnu -mcpu=pwr8 -mattr=+vsx < %s | FileCheck %s

define <4 x float> @bar(float* %p, float* %q) {
  %1 = bitcast float* %p to <12 x float>*
  %2 = bitcast float* %q to <12 x float>*
  %3 = load <12 x float>, <12 x float>* %1, align 16
  %4 = load <12 x float>, <12 x float>* %2, align 16
  %5 = fsub <12 x float> %4, %3
  %6 = shufflevector <12 x float> %5, <12 x float> undef, <4 x i32> <i32 0, i32 3, i32 6, i32 9>
  ret <4 x float>  %6

; CHECK: xxspltw
; CHECK-NEXT: xxspltw
; CHECK-NEXT: xxspltw
; CHECK-NEXT: vmrghw
; CHECK-NEXT: vmrghw
; CHECK-NEXT: xxswapd
; CHECK-NEXT: vsldoi
; CHECK-NEXT: blr
}

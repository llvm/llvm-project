; RUN: llc < %s -mcpu=cortex-a9 -mattr=+neon,+neonfp -float-abi=hard -mtriple armv7-linux-gnueabi | FileCheck %s

;; This test checks that the ExecutionDomainFix pass performs the domain changes
;; even when some dependencies are propagated through implicit definitions.

; CHECK: fun_a
define <4 x float> @fun_a(<4 x float> %in, <4 x float> %x, float %y) nounwind {
; CHECK: vext
; CHECK: vext
; CHECK: vadd.f32
  %1 = insertelement <4 x float> %in, float %y, i32 0
  %2 = fadd <4 x float> %1, %x  
  ret <4 x float> %2
}
; CHECK: fun_b
define <4 x i32> @fun_b(<4 x i32> %in, <4 x i32> %x, i32 %y) nounwind {
; CHECK: vmov.32
; CHECK: vadd.i32
  %1 = insertelement <4 x i32> %in, i32 %y, i32 0
  %2 = add <4 x i32> %1, %x  
  ret <4 x i32> %2
}

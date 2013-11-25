; RUN: llc < %s -mtriple=x86_64-linux-pc -mcpu=penryn | FileCheck %s

; Ensure PSRAD is generated as the condition is consumed by both PADD and
; BLENDVPS. PAND requires all bits setting properly.

define <4 x i32> @foo(<4 x i32>* %p, <4 x i1> %cond, <4 x i32> %v1, <4 x i32> %v2, <4 x i32> %v3) {
  %sext_cond = sext <4 x i1> %cond to <4 x i32>
  %t1 = add <4 x i32> %v1, %sext_cond
  %t2 = select <4 x i1> %cond, <4 x i32> %v1, <4 x i32> %v2
  store <4 x i32> %t2, <4 x i32>* %p
  ret <4 x i32> %t1
; CHECK: foo
; CHECK: pslld
; CHECK: psrad
; CHECK: ret
}

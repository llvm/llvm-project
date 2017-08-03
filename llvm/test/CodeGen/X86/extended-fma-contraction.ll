; RUN: llc -mcpu=bdver2 -mattr=-fma -mtriple=i686-apple-darwin < %s | FileCheck %s
; RUN: llc -mcpu=bdver2 -mattr=-fma,-fma4 -mtriple=i686-apple-darwin < %s | FileCheck %s --check-prefix=CHECK-NOFMA

; CHECK-LABEL: fmafunc
define <3 x float> @fmafunc(<3 x float> %a, <3 x float> %b, <3 x float> %c) {

; CHECK-NOT: vmulps
; CHECK-NOT: vaddps
; CHECK: vfmaddps
; CHECK-NOT: vmulps
; CHECK-NOT: vaddps

; CHECK-NOFMA-NOT: calll
; CHECK-NOFMA: vmulps
; CHECK-NOFMA: vaddps
; CHECK-NOFMA-NOT: calll

  %ret = tail call <3 x float> @llvm.fmuladd.v3f32(<3 x float> %a, <3 x float> %b, <3 x float> %c)
  ret <3 x float> %ret
}

declare <3 x float> @llvm.fmuladd.v3f32(<3 x float>, <3 x float>, <3 x float>) nounwind readnone

; RUN: llc -mtriple=arm-eabihf -mattr=+neon < %s | FileCheck -check-prefixes=CHECK-NO_NEON %s
; RUN: llc -mtriple=arm-eabihf -mattr=+neon,+neonfp < %s | FileCheck -check-prefixes=CHECK-NEON %s

define float @max_f32(float, float) {
;CHECK-NEON: vmax.f32
;CHECK-NO_NEON: vcmp.f32
;CHECK-NO_NEON: vmrs
;CHECK-NO_NEON: vmovgt.f32
  %3 = call nnan float @llvm.maxnum.f32(float %1, float %0)
  ret float %3
}

declare float @llvm.maxnum.f32(float, float) #1

define float @min_f32(float, float) {
;CHECK-NEON: vmin.f32
;CHECK-NO_NEON: vcmp.f32
;CHECK-NO_NEON: vmrs
;CHECK-NO_NEON: vmovlt.f32
  %3 = call nnan float @llvm.minnum.f32(float %1, float %0)
  ret float %3
}

declare float @llvm.minnum.f32(float, float) #1


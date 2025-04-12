; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_53 -mattr=+ptx42 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_53 -mattr=+ptx42 | %ptxas-verify -arch=sm_53 %}

declare half @llvm.nvvm.fma.rn.f16(half, half, half)
declare half @llvm.nvvm.fma.rn.ftz.f16(half, half, half)
declare half @llvm.nvvm.fma.rn.sat.f16(half, half, half)
declare half @llvm.nvvm.fma.rn.ftz.sat.f16(half, half, half)
declare <2 x half> @llvm.nvvm.fma.rn.f16x2(<2 x half>, <2 x half>, <2 x half>)
declare <2 x half> @llvm.nvvm.fma.rn.ftz.f16x2(<2 x half>, <2 x half>, <2 x half>)
declare <2 x half> @llvm.nvvm.fma.rn.sat.f16x2(<2 x half>, <2 x half>, <2 x half>)
declare <2 x half> @llvm.nvvm.fma.rn.ftz.sat.f16x2(<2 x half>, <2 x half>, <2 x half>)

; CHECK-LABEL: fma_rn_f16
define half @fma_rn_f16(half %0, half %1, half %2) {
  ; CHECK-NOT: call
  ; CHECK: fma.rn.f16
  %res = call half @llvm.nvvm.fma.rn.f16(half %0, half %1, half %2)
  ret half %res
}

; CHECK-LABEL: fma_rn_ftz_f16
define half @fma_rn_ftz_f16(half %0, half %1, half %2) {
  ; CHECK-NOT: call
  ; CHECK: fma.rn.ftz.f16
  %res = call half @llvm.nvvm.fma.rn.ftz.f16(half %0, half %1, half %2)
  ret half %res
}

; CHECK-LABEL: fma_rn_sat_f16
define half @fma_rn_sat_f16(half %0, half %1, half %2) {
  ; CHECK-NOT: call
  ; CHECK: fma.rn.sat.f16
  %res = call half @llvm.nvvm.fma.rn.sat.f16(half %0, half %1, half %2)
  ret half %res
}

; CHECK-LABEL: fma_rn_ftz_sat_f16
define half @fma_rn_ftz_sat_f16(half %0, half %1, half %2) {
  ; CHECK-NOT: call
  ; CHECK: fma.rn.ftz.sat.f16
  %res = call half @llvm.nvvm.fma.rn.ftz.sat.f16(half %0, half %1, half %2)
  ret half %res
}

; CHECK-LABEL: fma_rn_f16x2
define <2 x half> @fma_rn_f16x2(<2 x half> %0, <2 x half> %1, <2 x half> %2) {
  ; CHECK-NOT: call
  ; CHECK: fma.rn.f16x2
  %res = call <2 x half> @llvm.nvvm.fma.rn.f16x2(<2 x half> %0, <2 x half> %1, <2 x half> %2)
  ret <2 x half> %res
}

; CHECK-LABEL: fma_rn_ftz_f16x2
define <2 x half> @fma_rn_ftz_f16x2(<2 x half> %0, <2 x half> %1, <2 x half> %2) {
  ; CHECK-NOT: call
  ; CHECK: fma.rn.ftz.f16x2
  %res = call <2 x half> @llvm.nvvm.fma.rn.ftz.f16x2(<2 x half> %0, <2 x half> %1, <2 x half> %2)
  ret <2 x half> %res
}

; CHECK-LABEL: fma_rn_sat_f16x2
define <2 x half> @fma_rn_sat_f16x2(<2 x half> %0, <2 x half> %1, <2 x half> %2) {
  ; CHECK-NOT: call
  ; CHECK: fma.rn.sat.f16x2
  %res = call <2 x half> @llvm.nvvm.fma.rn.sat.f16x2(<2 x half> %0, <2 x half> %1, <2 x half> %2)
  ret <2 x half> %res
}

; CHECK-LABEL: fma_rn_ftz_sat_f16x2
define <2 x half> @fma_rn_ftz_sat_f16x2(<2 x half> %0, <2 x half> %1, <2 x half> %2) {
  ; CHECK-NOT: call
  ; CHECK: fma.rn.ftz.sat.f16x2
  %res = call <2 x half> @llvm.nvvm.fma.rn.ftz.sat.f16x2(<2 x half> %0, <2 x half> %1, <2 x half> %2)
  ret <2 x half> %res
}

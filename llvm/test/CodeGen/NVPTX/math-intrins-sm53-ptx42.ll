; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_53 -mattr=+ptx42 | FileCheck %s
; RUN: %if ptxas-sm_53 && ptxas-isa-4.2 %{ llc < %s -mtriple=nvptx64 -mcpu=sm_53 -mattr=+ptx42 | %ptxas-verify -arch=sm_53 %}

declare half @llvm.nvvm.fma.f16(half, half, half, i32)
declare half @llvm.nvvm.fma.ftz.f16(half, half, half, i32)
declare half @llvm.nvvm.fma.sat.f16(half, half, half, i32)
declare half @llvm.nvvm.fma.ftz.sat.f16(half, half, half, i32)
declare <2 x half> @llvm.nvvm.fma.v2f16(<2 x half>, <2 x half>, <2 x half>, i32)
declare <2 x half> @llvm.nvvm.fma.ftz.v2f16(<2 x half>, <2 x half>, <2 x half>, i32)
declare <2 x half> @llvm.nvvm.fma.sat.v2f16(<2 x half>, <2 x half>, <2 x half>, i32)
declare <2 x half> @llvm.nvvm.fma.ftz.sat.v2f16(<2 x half>, <2 x half>, <2 x half>, i32)

; CHECK-LABEL: fma_rn_f16
define half @fma_rn_f16(half %0, half %1, half %2) {
  ; CHECK-NOT: call
  ; CHECK: fma.rn.f16
  %res = call half @llvm.nvvm.fma.f16(half %0, half %1, half %2, /* rnd=rn */ i32 1)
  ret half %res
}

; CHECK-LABEL: fma_rn_ftz_f16
define half @fma_rn_ftz_f16(half %0, half %1, half %2) {
  ; CHECK-NOT: call
  ; CHECK: fma.rn.ftz.f16
  %res = call half @llvm.nvvm.fma.ftz.f16(half %0, half %1, half %2, /* rnd=rn */ i32 1)
  ret half %res
}

; CHECK-LABEL: fma_rn_sat_f16
define half @fma_rn_sat_f16(half %0, half %1, half %2) {
  ; CHECK-NOT: call
  ; CHECK: fma.rn.sat.f16
  %res = call half @llvm.nvvm.fma.sat.f16(half %0, half %1, half %2, /* rnd=rn */ i32 1)
  ret half %res
}

; CHECK-LABEL: fma_rn_ftz_sat_f16
define half @fma_rn_ftz_sat_f16(half %0, half %1, half %2) {
  ; CHECK-NOT: call
  ; CHECK: fma.rn.ftz.sat.f16
  %res = call half @llvm.nvvm.fma.ftz.sat.f16(half %0, half %1, half %2, /* rnd=rn */ i32 1)
  ret half %res
}

; CHECK-LABEL: fma_rn_f16x2
define <2 x half> @fma_rn_f16x2(<2 x half> %0, <2 x half> %1, <2 x half> %2) {
  ; CHECK-NOT: call
  ; CHECK: fma.rn.f16x2
  %res = call <2 x half> @llvm.nvvm.fma.v2f16(<2 x half> %0, <2 x half> %1, <2 x half> %2, /* rnd=rn */ i32 1)
  ret <2 x half> %res
}

; CHECK-LABEL: fma_rn_ftz_f16x2
define <2 x half> @fma_rn_ftz_f16x2(<2 x half> %0, <2 x half> %1, <2 x half> %2) {
  ; CHECK-NOT: call
  ; CHECK: fma.rn.ftz.f16x2
  %res = call <2 x half> @llvm.nvvm.fma.ftz.v2f16(<2 x half> %0, <2 x half> %1, <2 x half> %2, /* rnd=rn */ i32 1)
  ret <2 x half> %res
}

; CHECK-LABEL: fma_rn_sat_f16x2
define <2 x half> @fma_rn_sat_f16x2(<2 x half> %0, <2 x half> %1, <2 x half> %2) {
  ; CHECK-NOT: call
  ; CHECK: fma.rn.sat.f16x2
  %res = call <2 x half> @llvm.nvvm.fma.sat.v2f16(<2 x half> %0, <2 x half> %1, <2 x half> %2, /* rnd=rn */ i32 1)
  ret <2 x half> %res
}

; CHECK-LABEL: fma_rn_ftz_sat_f16x2
define <2 x half> @fma_rn_ftz_sat_f16x2(<2 x half> %0, <2 x half> %1, <2 x half> %2) {
  ; CHECK-NOT: call
  ; CHECK: fma.rn.ftz.sat.f16x2
  %res = call <2 x half> @llvm.nvvm.fma.ftz.sat.v2f16(<2 x half> %0, <2 x half> %1, <2 x half> %2, /* rnd=rn */ i32 1)
  ret <2 x half> %res
}

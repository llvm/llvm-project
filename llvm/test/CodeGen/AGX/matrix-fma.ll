; RUN: llc -mtriple=agx -mcpu=agx-g13g -o /dev/null -print-after=agx-codegenprepare < %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g13g -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM

; The 8x8 SIMD-group matrix multiply-accumulate gathers the row of A and the
; two columns of B of each thread with 24 shuffles and accumulates with 16
; fused multiply-adds.
; CHECK-LABEL: @mma(
; CHECK-NOT: llvm.agx.simdgroup.matrix.fma
; CHECK-COUNT-24: call float @llvm.agx.simd.shuffle.f32
; CHECK-COUNT-16: call float @llvm.fma.f32
; ASM-LABEL: mma:
; ASM: simd_shuffle
; ASM: fmadd32
define agx_kernel void @mma(ptr addrspace(1) %o, <2 x float> %a, <2 x float> %b, <2 x float> %c) {
  %r = call <2 x float> @llvm.agx.simdgroup.matrix.fma.v2f32(<2 x float> %a, <2 x float> %b, <2 x float> %c)
  store <2 x float> %r, ptr addrspace(1) %o
  ret void
}

declare <2 x float> @llvm.agx.simdgroup.matrix.fma.v2f32(<2 x float>, <2 x float>, <2 x float>)

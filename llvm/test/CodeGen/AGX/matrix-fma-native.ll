; RUN: llc -mtriple=agx -mcpu=agx-g14c -o /dev/null -print-after=agx-codegenprepare < %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs -show-mc-encoding < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=agx -mcpu=agx-g13g -o /dev/null -print-after=agx-codegenprepare < %s 2>&1 | FileCheck %s --check-prefix=G13

; On G14 the matrix unit computes the 8x8 SIMD-group multiply-accumulate with
; the same results as the shuffled form (an in-order chain of fused
; multiply-adds): simd_matrix_fmadd32 for float, simd_matrix_fmadd16 for
; half with IEEE denormals. Float with ieee-fp32 IEEE denormals, half that
; flushes, canonical NaNs and bfloat keep the shuffles.

; CHECK-LABEL: @mma32(
; CHECK: call i64 @llvm.agx.simd.matrix.fmadd32(i64
; CHECK-NOT: shuffle
; ASM-LABEL: mma32:
; The pairs are encoded as their first register (the bytes of applegpu's
; assembler for the same operands).
; ASM: simd_matrix_fmadd32 r2_r3, r6_r7, r4_r5, r2_r3 // encoding: [0x6f,0x09,0x4c,0x8e,0x24,0x44,0x02,0x80]
; G13-LABEL: @mma32(
; G13-COUNT-24: call float @llvm.agx.simd.shuffle.f32
define agx_kernel void @mma32(ptr addrspace(1) %o, <2 x float> %a, <2 x float> %b, <2 x float> %c) {
  %r = call <2 x float> @llvm.agx.simdgroup.matrix.fma.v2f32(<2 x float> %a, <2 x float> %b, <2 x float> %c)
  store <2 x float> %r, ptr addrspace(1) %o
  ret void
}

; CHECK-LABEL: @mma16(
; CHECK: call i32 @llvm.agx.simd.matrix.fmadd16(i32
; CHECK-NOT: shuffle
; ASM-LABEL: mma16:
; ASM: simd_matrix_fmadd16 r1, r3, r2, r1 // encoding: [0x6f,0x04,0x46,0x48,0x04,0x42,0x00,0x80]
define agx_kernel void @mma16(ptr addrspace(1) %o, <2 x half> %a, <2 x half> %b, <2 x half> %c) #0 {
  %r = call <2 x half> @llvm.agx.simdgroup.matrix.fma.v2f16(<2 x half> %a, <2 x half> %b, <2 x half> %c)
  store <2 x half> %r, ptr addrspace(1) %o
  ret void
}

; CHECK-LABEL: @mma16_flush(
; CHECK-NOT: matrix.fmadd
; CHECK: call half @llvm.agx.simd.shuffle.f16
define agx_kernel void @mma16_flush(ptr addrspace(1) %o, <2 x half> %a, <2 x half> %b, <2 x half> %c) #1 {
  %r = call <2 x half> @llvm.agx.simdgroup.matrix.fma.v2f16(<2 x half> %a, <2 x half> %b, <2 x half> %c)
  store <2 x half> %r, ptr addrspace(1) %o
  ret void
}

; CHECK-LABEL: @mma32_ieee(
; CHECK-NOT: matrix.fmadd
; CHECK: call float @llvm.agx.simd.shuffle.f32
define agx_kernel void @mma32_ieee(ptr addrspace(1) %o, <2 x float> %a, <2 x float> %b, <2 x float> %c) #2 {
  %r = call <2 x float> @llvm.agx.simdgroup.matrix.fma.v2f32(<2 x float> %a, <2 x float> %b, <2 x float> %c)
  store <2 x float> %r, ptr addrspace(1) %o
  ret void
}

; CHECK-LABEL: @mma32_canonical(
; CHECK-NOT: matrix.fmadd
; CHECK: call float @llvm.agx.simd.shuffle.f32
define agx_kernel void @mma32_canonical(ptr addrspace(1) %o, <2 x float> %a, <2 x float> %b, <2 x float> %c) #3 {
  %r = call <2 x float> @llvm.agx.simdgroup.matrix.fma.v2f32(<2 x float> %a, <2 x float> %b, <2 x float> %c)
  store <2 x float> %r, ptr addrspace(1) %o
  ret void
}

declare <2 x float> @llvm.agx.simdgroup.matrix.fma.v2f32(<2 x float>, <2 x float>, <2 x float>)
declare <2 x half> @llvm.agx.simdgroup.matrix.fma.v2f16(<2 x half>, <2 x half>, <2 x half>)

attributes #0 = { denormal_fpenv(ieee) }
attributes #1 = { denormal_fpenv(preservesign) }
attributes #2 = { denormal_fpenv(float: ieee) "target-features"="+ieee-fp32" }
attributes #3 = { "target-features"="+canonical-nan" }

; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s

; Two loads into the halves of one register pair (a matrix fragment read
; transposed, its elements a row apart) run together: the implicit use and def
; of the pair that keep the other half live are not a dependence on the first
; load. Only the matrix multiply-add waits, once, for both.

; CHECK-LABEL: fragment:
; CHECK: device_load {{[01]}}, i32, x, r{{[0-9]+}},
; CHECK-NOT: wait
; CHECK: device_load {{[01]}}, i32, x, r{{[0-9]+}},
; CHECK: device_load {{[01]}}, i32, xy,
; CHECK: wait
; CHECK-NOT: wait
; CHECK: simd_matrix_fmadd32
define void @fragment(ptr addrspace(1) %a, ptr addrspace(1) %b, i64 %ld,
                      ptr addrspace(1) %out) {
  %p1 = getelementptr float, ptr addrspace(1) %a, i64 %ld
  %x = load float, ptr addrspace(1) %a, align 4
  %y = load float, ptr addrspace(1) %p1, align 4
  %v0 = insertelement <2 x float> poison, float %x, i64 0
  %v = insertelement <2 x float> %v0, float %y, i64 1
  %w = load <2 x float>, ptr addrspace(1) %b, align 8
  %r = call <2 x float> @llvm.agx.simdgroup.matrix.fma.v2f32(<2 x float> %v, <2 x float> %w, <2 x float> zeroinitializer)
  store <2 x float> %r, ptr addrspace(1) %out, align 8
  ret void
}

declare <2 x float> @llvm.agx.simdgroup.matrix.fma.v2f32(<2 x float>, <2 x float>, <2 x float>)

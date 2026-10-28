; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s

; minimumnum / maximumnum: the hardware's NaN-ignoring min / max, and for
; operands that compare equal the bits ORed / ANDed (-0 below +0).

; CHECK-LABEL: max:
; CHECK-DAG:   fcmpsel gtn, [[M:r[0-9]+]], u2, u3, u2, u3
; CHECK-DAG:   and [[Z:r[0-9]+]], u2, u3
; CHECK:       fcmpsel eq, [[R:r[0-9]+]], u2, u3, [[Z]], [[M]]
; CHECK-NEXT:  device_store 0, i32, x, [[R]],
define agx_kernel void @max(ptr addrspace(1) %p, float %a, float %b) {
  %m = call float @llvm.maximumnum.f32(float %a, float %b)
  store float %m, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: min_half:
; CHECK-DAG:   fcmpsel ltn, [[M:r[0-9]+[lh]]],
; CHECK-DAG:   or [[Z:r[0-9]+[lh]]],
; CHECK:       fcmpsel eq, {{r[0-9]+[lh]}}, {{.*}}, [[Z]], [[M]]
define agx_kernel void @min_half(ptr addrspace(1) %p, half %a, half %b) {
  %m = call half @llvm.minimumnum.f16(half %a, half %b)
  store half %m, ptr addrspace(1) %p
  ret void
}

; Without signed zeros, the hardware min / max alone.
; CHECK-LABEL: max_nsz:
; CHECK:       fcmpsel gtn, [[R:r[0-9]+]], u2, u3, u2, u3
; CHECK-NEXT:  device_store 0, i32, x, [[R]],
define agx_kernel void @max_nsz(ptr addrspace(1) %p, float %a, float %b) {
  %m = call nsz float @llvm.maximumnum.f32(float %a, float %b)
  store float %m, ptr addrspace(1) %p
  ret void
}

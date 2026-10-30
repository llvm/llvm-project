; powi with a variable exponent is pow(|x|, n) with the sign of x for an odd
; n (SelectionDAG's fallback, pow(x, n), is NaN for a negative x); a
; constant exponent is left to SelectionDAG's multiplications. The precise
; pow is NaN for a negative finite x and a non-integer y only: pow(-inf, 0.5)
; is +inf.
; RUN: llc -mtriple=agx -mcpu=agx-g14g -stop-after=agx-codegenprepare %s -o - | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14g %s -o /dev/null

; CHECK-LABEL: @powi_afn(
; CHECK-NOT: @llvm.powi
; CHECK: %[[ABS:.*]] = call afn float @llvm.fabs.f32(float %x)
; CHECK: %[[Y:.*]] = sitofp afn i32 %n to float
; CHECK: %[[MAG:.*]] = call afn float @llvm.pow.f32(float %[[ABS]], float %[[Y]])
; CHECK: %[[ODD:.*]] = trunc i32 %n to i1
; CHECK: %[[SIGN:.*]] = select afn i1 %[[ODD]], float %x, float 1.000000e+00
; CHECK: %r = call afn float @llvm.copysign.f32(float %[[MAG]], float %[[SIGN]])
define agx_kernel void @powi_afn(ptr addrspace(1) %p, float %x, i32 %n) {
  %r = call afn float @llvm.powi.f32.i32(float %x, i32 %n)
  store float %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: @powi_half(
; CHECK: fpext half %x to float
; CHECK-NOT: @llvm.pow.f32
; CHECK: call float @llvm.copysign.f32
; CHECK: fptrunc float %{{.*}} to half
define agx_kernel void @powi_half(ptr addrspace(1) %p, half %x, i32 %n) {
  %r = call half @llvm.powi.f16.i32(half %x, i32 %n)
  store half %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: @powi_const(
; CHECK: call float @llvm.powi.f32.i32(float %x, i32 3)
define agx_kernel void @powi_const(ptr addrspace(1) %p, float %x) {
  %r = call float @llvm.powi.f32.i32(float %x, i32 3)
  store float %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: @pow(
; CHECK: %[[NEG:.*]] = fcmp olt float %x, 0.000000e+00
; CHECK-NEXT: %[[FINITE:.*]] = fcmp ogt float %x, -inf
; CHECK-NEXT: and i1 %[[NEG]], %[[FINITE]]
define agx_kernel void @pow(ptr addrspace(1) %p, float %x, float %y) {
  %r = call float @llvm.pow.f32(float %x, float %y)
  store float %r, ptr addrspace(1) %p
  ret void
}

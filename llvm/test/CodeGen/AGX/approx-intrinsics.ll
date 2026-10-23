; RUN: opt -passes='default<O2>' -mtriple=agx -mcpu=agx-g14c -S < %s | FileCheck %s --check-prefix=IR
; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -mattr=+ieee-fp32 -stop-after=agx-strict-fp < %s | FileCheck %s --check-prefix=IEEE

; llvm.agx.sin and the others are the hardware's approximations: LLVM keeps
; them for constant arguments (llvm.sin of a constant is folded with the
; host's libm), so a constant argument gives the bits a variable one does.

; IR-LABEL: @constant(
; IR: {{(tail )?}}call float @llvm.agx.sin.f32(float 1.000000e+00)
; IR: {{(tail )?}}call float @llvm.agx.cos.f32(float 1.000000e+00)
; IR: {{(tail )?}}call float @llvm.agx.exp2.f32(float 5.000000e-01)
; IR: {{(tail )?}}call float @llvm.agx.log2.f32(float 3.000000e+00)
define agx_kernel void @constant(ptr addrspace(1) %out) {
  %s = call float @llvm.agx.sin.f32(float 1.0)
  %c = call float @llvm.agx.cos.f32(float 1.0)
  %e = call float @llvm.agx.exp2.f32(float 0.5)
  %l = call float @llvm.agx.log2.f32(float 3.0)
  store float %s, ptr addrspace(1) %out
  %p1 = getelementptr float, ptr addrspace(1) %out, i32 1
  store float %c, ptr addrspace(1) %p1
  %p2 = getelementptr float, ptr addrspace(1) %out, i32 2
  store float %e, ptr addrspace(1) %p2
  %p3 = getelementptr float, ptr addrspace(1) %out, i32 3
  store float %l, ptr addrspace(1) %p3
  ret void
}

; The same instructions as llvm.sin, llvm.cos, llvm.exp2 and llvm.log2.
; CHECK-LABEL: variable:
; CHECK: sin_pt_1
; CHECK: sin_pt_2
; CHECK: sin_pt_1
; CHECK: sin_pt_2
; CHECK: exp2
; CHECK: log2
define agx_kernel void @variable(ptr addrspace(1) %out, float %a) {
  %s = call float @llvm.agx.sin.f32(float %a)
  %c = call float @llvm.agx.cos.f32(float %s)
  %e = call float @llvm.agx.exp2.f32(float %c)
  %l = call float @llvm.agx.log2.f32(float %e)
  store float %l, ptr addrspace(1) %out
  ret void
}

; With ieee-fp32, in a function that keeps denormals, they get the IEEE lowering of llvm.sin
; and the others: x below 2^-12 gives x, 2^x is 2^f * 2^floor(x) with the
; fraction's power from the hardware, denormal log2 operands are scaled.
; IEEE-LABEL: define agx_kernel void @ieee(
; IEEE:         %s = call float @llvm.agx.sin.f32(float %a)
; IEEE:         fcmp olt float {{.*}}, f0x39800000
; IEEE:         call float @llvm.floor.f32(
; IEEE:         call float @llvm.agx.exp2.f32(
; IEEE:         call float @llvm.agx.log2.f32(
define agx_kernel void @ieee(ptr addrspace(1) %out, float %a) {
  %s = call float @llvm.agx.sin.f32(float %a)
  %e = call float @llvm.agx.exp2.f32(float %s)
  %l = call float @llvm.agx.log2.f32(float %e)
  store float %l, ptr addrspace(1) %out
  ret void
}

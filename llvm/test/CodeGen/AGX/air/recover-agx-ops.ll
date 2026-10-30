; agx-lower-air compiles AIR's portable forms of operations AGX has as those
; operations: afn 1 / x, 1 / sqrt(x) and sin / cos of x * pi / 2 as the
; hardware approximations, AIR's minimum (fmin with a NaN check) as AGX's,
; in the narrow type when computed in float, and drops PTX's explicit .ftz
; flushes where the f32 ALU flushes anyway.
; RUN: llc -mtriple=air -air-portable %s -o %t.air.ll
; RUN: opt -mtriple=agx -mcpu=agx-g14g -passes=agx-lower-air -S %t.air.ll | FileCheck %s

; CHECK-LABEL: define agx_kernel void @k(
; CHECK: call float @llvm.agx.rcp.f32(float %x.val)
; CHECK: call float @llvm.agx.rsqrt.f32(float %x.val)
; CHECK: call float @llvm.agx.sin.f32(float %x.val)
; CHECK: call float @llvm.minimum.f32(float %x.val, float %y.val)
; CHECK: call bfloat @llvm.minimum.bf16(bfloat %b.val, bfloat %c.val)
; CHECK: %ff = fadd float %x.val, %y.val
; CHECK-NOT: copysign
; CHECK: ret void
define void @k(ptr addrspace(1) %out, float %x, float %y, bfloat %b, bfloat %c) "air-kernel" "denormal-fp-math-f32"="preservesign,preservesign" {
  %r = fdiv afn float 1.0, %x
  %s = call afn float @llvm.sqrt.f32(float %x)
  %rs = fdiv afn float 1.0, %s
  %q = fmul afn float %x, 0x3FF921FB60000000
  %sin = call afn float @llvm.sin.f32(float %q)
  %m = call float @llvm.minimum.f32(float %x, float %y)
  %be = fpext bfloat %b to float
  %ce = fpext bfloat %c to float
  %bm = call float @llvm.minimum.f32(float %be, float %ce)
  %bt = fptrunc float %bm to bfloat
  ; add.ftz: the result flushed explicitly
  %f = fadd float %x, %y
  %a = call float @llvm.fabs.f32(float %f)
  %t = fcmp olt float %a, 0x3810000000000000
  %z = call float @llvm.copysign.f32(float 0.0, float %f)
  %ff = select i1 %t, float %z, float %f
  store volatile float %r, ptr addrspace(1) %out
  store volatile float %rs, ptr addrspace(1) %out
  store volatile float %sin, ptr addrspace(1) %out
  store volatile float %m, ptr addrspace(1) %out
  store volatile bfloat %bt, ptr addrspace(1) %out
  store volatile float %ff, ptr addrspace(1) %out
  ret void
}

; An afn division of anything but 1 stays a division (it was replaced by its
; divisor).
; CHECK-LABEL: define agx_kernel void @div(
; CHECK: %r = fdiv afn float %x.val, %y.val
; CHECK-NEXT: store float %r,
define void @div(ptr addrspace(1) %out, float %x, float %y) "air-kernel" {
  %r = fdiv afn float %x, %y
  store float %r, ptr addrspace(1) %out
  ret void
}

declare float @llvm.sqrt.f32(float)
declare float @llvm.sin.f32(float)
declare float @llvm.minimum.f32(float, float)
declare float @llvm.fabs.f32(float)
declare float @llvm.copysign.f32(float, float)

; RUN: llc -mtriple=agx -mcpu=agx-g13g -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g13g -mattr=+ieee-fp32 -o /dev/null -print-after=agx-strict-fp < %s 2>&1 | FileCheck %s --check-prefix=STRICT
; RUN: llc -mtriple=agx -mcpu=agx-g13g -o /dev/null -print-after=agx-lower-fp-env < %s 2>&1 | FileCheck %s --check-prefix=ENV

; A negation or absolute value of its own is a bit operation, which keeps
; denormals.
; CHECK-LABEL: neg:
; CHECK: xor
; CHECK-NOT: fadd
define agx_kernel void @neg(ptr addrspace(1) %o, float %a) #0 {
  %n = fneg float %a
  store float %n, ptr addrspace(1) %o
  ret void
}

; CHECK-LABEL: abs:
; CHECK: and
; CHECK-NOT: fadd
define agx_kernel void @abs(ptr addrspace(1) %o, float %a) #0 {
  %n = call float @llvm.fabs.f32(float %a)
  store float %n, ptr addrspace(1) %o
  ret void
}

; The hardware minimum keeps a denormal operand: its result is flushed in a
; function that flushes float denormals.
; CHECK-LABEL: min_ftz:
; CHECK: fcmpsel
; CHECK-NEXT: fmul32 {{.*}}, 1.0
define agx_kernel void @min_ftz(ptr addrspace(1) %o, float %a, float %b) #0 {
  %m = call float @llvm.minnum.f32(float %a, float %b)
  store float %m, ptr addrspace(1) %o
  ret void
}

; float to bfloat rounds to nearest even.
; CHECK-LABEL: to_bf16:
; CHECK: iadd
; CHECK: 32767
define agx_kernel void @to_bf16(ptr addrspace(1) %o, float %a) {
  %b = fptrunc float %a to bfloat
  store bfloat %b, ptr addrspace(1) %o
  ret void
}

; With ieee-fp32, bfloat arithmetic is float arithmetic on the extended
; operands, tagged for the IEEE lowering, rounded back once; fma is computed
; exactly with integer arithmetic.
; STRICT-LABEL: @bf16_add(
; STRICT: fpext bfloat %a to float
; STRICT: fptrunc float {{.*}} to bfloat
; STRICT-NOT: fadd bfloat
define agx_kernel void @bf16_add(ptr addrspace(1) %o, bfloat %a, bfloat %b) {
  %s = fadd bfloat %a, %b
  store bfloat %s, ptr addrspace(1) %o
  ret void
}

; STRICT-LABEL: @bf16_fma(
; STRICT-NOT: @llvm.fma
; STRICT: mul i128
define agx_kernel void @bf16_fma(ptr addrspace(1) %o, bfloat %a, bfloat %b, bfloat %c) {
  %s = call bfloat @llvm.fma.bf16(bfloat %a, bfloat %b, bfloat %c)
  store bfloat %s, ptr addrspace(1) %o
  ret void
}

; Vector constrained operations are rounded lane by lane.
; ENV-LABEL: @vector_rz(
; ENV-NOT: call {{.*}}constrained
; ENV: insertelement <2 x float>
define agx_kernel void @vector_rz(ptr addrspace(1) %o, <2 x float> %a, <2 x float> %b) #1 {
  %s = call <2 x float> @llvm.experimental.constrained.fadd.v2f32(<2 x float> %a, <2 x float> %b, metadata !"round.towardzero", metadata !"fpexcept.ignore") #1
  store <2 x float> %s, ptr addrspace(1) %o
  ret void
}

declare float @llvm.fabs.f32(float)
declare float @llvm.minnum.f32(float, float)
declare bfloat @llvm.fma.bf16(bfloat, bfloat, bfloat)
declare <2 x float> @llvm.experimental.constrained.fadd.v2f32(<2 x float>, <2 x float>, metadata, metadata)

attributes #0 = { denormal_fpenv(preservesign) }
attributes #1 = { strictfp }

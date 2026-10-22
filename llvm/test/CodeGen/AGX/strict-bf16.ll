; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s

; Constrained bfloat16 arithmetic is promoted to f32 and narrowed with
; STRICT_FP_TO_BF16, rounded to nearest even like FP_TO_BF16:
; (x + 0x7fff + lsb) >> 16, with NaNs quieted.

; CHECK-LABEL: strict_fadd_bf16:
; CHECK: mov_imm {{r[0-9]+}}, 32767
; CHECK: fadd32
; CHECK: icmpsel ugt
; CHECK: device_store 0, i16
define agx_kernel void @strict_fadd_bf16(ptr addrspace(1) %o, bfloat %a, bfloat %b) strictfp {
  %r = call bfloat @llvm.experimental.constrained.fadd.bf16(bfloat %a, bfloat %b, metadata !"round.dynamic", metadata !"fpexcept.ignore") strictfp
  store bfloat %r, ptr addrspace(1) %o
  ret void
}

; CHECK-LABEL: strict_fadd_v2bf16:
; CHECK-COUNT-2: fadd32
define agx_kernel void @strict_fadd_v2bf16(ptr addrspace(1) %o, <2 x bfloat> %a, <2 x bfloat> %b) strictfp {
  %r = call <2 x bfloat> @llvm.experimental.constrained.fadd.v2bf16(<2 x bfloat> %a, <2 x bfloat> %b, metadata !"round.dynamic", metadata !"fpexcept.ignore") strictfp
  store <2 x bfloat> %r, ptr addrspace(1) %o
  ret void
}

declare bfloat @llvm.experimental.constrained.fadd.bf16(bfloat, bfloat, metadata, metadata)
declare <2 x bfloat> @llvm.experimental.constrained.fadd.v2bf16(<2 x bfloat>, <2 x bfloat>, metadata, metadata)

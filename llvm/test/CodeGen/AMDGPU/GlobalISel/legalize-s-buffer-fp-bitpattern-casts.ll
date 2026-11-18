; RUN: llc -global-isel -mtriple=amdgpu12.50 -stop-after=legalizer -verify-machineinstrs -o - %s | FileCheck %s

define amdgpu_ps half @s_buffer_load_half(<4 x i32> inreg %rsrc) {
; CHECK-LABEL: name: s_buffer_load_half
; CHECK: [[LOAD:%[0-9]+]]:_(i32) = G_AMDGPU_S_BUFFER_LOAD_USHORT
; CHECK-NEXT: [[TRUNC:%[0-9]+]]:_(i16) = G_TRUNC [[LOAD]](i32)
; CHECK-NEXT: {{%[0-9]+}}:_(f16) = G_BITCAST [[TRUNC]](i16)
  %x = call half @llvm.amdgcn.s.buffer.load.f16(<4 x i32> %rsrc, i32 0, i32 0)
  ret half %x
}

define amdgpu_ps bfloat @s_buffer_load_bfloat(<4 x i32> inreg %rsrc) {
; CHECK-LABEL: name: s_buffer_load_bfloat
; CHECK: [[LOAD:%[0-9]+]]:_(i32) = G_AMDGPU_S_BUFFER_LOAD_USHORT
; CHECK-NEXT: [[TRUNC:%[0-9]+]]:_(i16) = G_TRUNC [[LOAD]](i32)
; CHECK-NEXT: {{%[0-9]+}}:_(bf16) = G_BITCAST [[TRUNC]](i16)
  %x = call bfloat @llvm.amdgcn.s.buffer.load.bf16(<4 x i32> %rsrc, i32 0, i32 0)
  ret bfloat %x
}

declare half @llvm.amdgcn.s.buffer.load.f16(<4 x i32>, i32, i32 immarg)
declare bfloat @llvm.amdgcn.s.buffer.load.bf16(<4 x i32>, i32, i32 immarg)

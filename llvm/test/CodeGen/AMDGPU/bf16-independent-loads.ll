; RUN: llc -mtriple=amdgpu11.00-amd-amdhsa -mcpu=gfx1100 -O3 -verify-machineinstrs %s -o - | FileCheck %s
;
; The shared low half must not cause the coalescer to join the two VMEM
; results before scheduling. Preserve the d16_hi load folding.
; CHECK-LABEL: silu:
; CHECK: global_load_d16_hi_b16
; CHECK-NOT: v_exp_f32
; CHECK: global_load_d16_hi_b16
; CHECK: v_exp_f32
; CHECK: global_store

target triple = "amdgpu11.00-amd-amdhsa"
define amdgpu_kernel void @silu(ptr addrspace(1) %x, ptr addrspace(1) %gate, ptr addrspace(1) %out) {
  %i = call i32 @llvm.amdgcn.workitem.id.x()
  %xp = getelementptr bfloat, ptr addrspace(1) %x, i32 %i
  %gp = getelementptr bfloat, ptr addrspace(1) %gate, i32 %i
  %op = getelementptr bfloat, ptr addrspace(1) %out, i32 %i
  %xb = load bfloat, ptr addrspace(1) %xp, align 2
  %xpack = insertelement <2 x bfloat> <bfloat 0.0, bfloat poison>, bfloat %xb, i32 1
  %xf = bitcast <2 x bfloat> %xpack to float
  %scaled = fmul float %xf, 0xBFF7154760000000
  %ex = call float @llvm.amdgcn.exp2.f32(float %scaled)
  %denom = fadd float %ex, 1.0
  %sig = fdiv float 1.0, %denom
  %silu = fmul float %xf, %sig
  %gb = load bfloat, ptr addrspace(1) %gp, align 2
  %gpack = insertelement <2 x bfloat> <bfloat 0.0, bfloat poison>, bfloat %gb, i32 1
  %gf = bitcast <2 x bfloat> %gpack to float
  %result = fmul float %silu, %gf
  %bf = fptrunc float %result to bfloat
  store bfloat %bf, ptr addrspace(1) %op, align 2
  ret void
}
declare i32 @llvm.amdgcn.workitem.id.x()
declare float @llvm.amdgcn.exp2.f32(float)

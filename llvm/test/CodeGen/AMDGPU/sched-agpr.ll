; RUN: llc -mtriple=amdgcn -mcpu=gfx942 < %s | FileCheck -check-prefix=GCN %s

; sched_agpr's regnum is an architectural AGPR index (aN), not a unified-file
; offset. vN and aN share a HW index, so a strict hint must also constrain the
; register file. The strict pseudo stays through coalescing (its AGPR operand
; class blocks inflation to AV_*) and is expanded after RA.

declare i32 @llvm.amdgcn.sched.agpr.i32(i32, i32, i32)
declare <4 x float> @llvm.amdgcn.sched.agpr.v4f32(<4 x float>, i32, i32)

; --------------------------------------------------------------------------
; Result is only used by VALU. strict hint 8 still lands on a8; VALU reads it
; back into a VGPR.

; GCN-LABEL: {{^}}agpr8_as_vgpr:
; GCN: v_accvgpr_write_b32 a8,
; GCN: v_accvgpr_read_b32 v{{[0-9]+}}, a8
; GCN: v_add_u32_e32
; GCN: .Lagpr8_as_vgpr.num_agpr, 9
define amdgpu_kernel void @agpr8_as_vgpr(ptr addrspace(1) %out, i32 %x) #0 {
  %h = call i32 @llvm.amdgcn.sched.agpr.i32(i32 %x, i32 8, i32 1)
  %y = add i32 %h, 1
  store i32 %y, ptr addrspace(1) %out
  ret void
}

; --------------------------------------------------------------------------
; Same hint, but the result is consumed as an AGPR. Lands on a8.

; GCN-LABEL: {{^}}agpr8_kept:
; GCN: v_accvgpr_write_b32 a8,
; GCN: .Lagpr8_kept.num_agpr, 9
define amdgpu_kernel void @agpr8_kept(i32 %x) #0 {
  %h = call i32 @llvm.amdgcn.sched.agpr.i32(i32 %x, i32 8, i32 1)
  call void asm sideeffect "; use $0", "a"(i32 %h)
  ret void
}

; --------------------------------------------------------------------------
; 128-bit value, strict hint 4, AGPR use. Starting index is a4: a[4:7].

; GCN-LABEL: {{^}}agpr4_v4:
; GCN-DAG: v_accvgpr_write_b32 a4,
; GCN-DAG: v_accvgpr_write_b32 a5,
; GCN-DAG: v_accvgpr_write_b32 a6,
; GCN-DAG: v_accvgpr_write_b32 a7,
; GCN: .Lagpr4_v4.num_agpr, 8
define amdgpu_kernel void @agpr4_v4(ptr addrspace(1) %in) #0 {
  %x = load <4 x float>, ptr addrspace(1) %in
  %h = call <4 x float> @llvm.amdgcn.sched.agpr.v4f32(<4 x float> %x, i32 4, i32 1)
  call void asm sideeffect "; use $0", "a"(<4 x float> %h)
  ret void
}

; --------------------------------------------------------------------------
; regnum 0 is a real a0, not "no hint".

; GCN-LABEL: {{^}}agpr0_kept:
; GCN: v_accvgpr_write_b32 a0,
; GCN: .Lagpr0_kept.num_agpr, 1
define amdgpu_kernel void @agpr0_kept(i32 %x) #0 {
  %h = call i32 @llvm.amdgcn.sched.agpr.i32(i32 %x, i32 0, i32 1)
  call void asm sideeffect "; use $0", "a"(i32 %h)
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="64,64" "amdgpu-waves-per-eu"="1,1" }

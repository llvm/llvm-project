; RUN: llc -mtriple=amdgcn -mcpu=gfx1250 < %s | FileCheck -check-prefix=GCN %s

; Verify that llvm.amdgcn.schedule.{vgpr,sgpr} steer register allocation to
; the exact requested register number. Kernels are pinned to occupancy 1
; (waves-per-eu=1,1 with a single-wave workgroup) so the whole 1024-VGPR file
; is addressable and high-numbered registers are available to the allocator.

declare i32 @llvm.amdgcn.sched.vgpr.i32(i32, i32, i32)
declare i64 @llvm.amdgcn.sched.vgpr.i64(i64, i32, i32)
declare <16 x float> @llvm.amdgcn.sched.vgpr.v16f32(<16 x float>, i32, i32)

; --------------------------------------------------------------------------
; sched_vgpr: strict hint to v42

; GCN-LABEL: {{^}}vgpr42:
; GCN: v42
define amdgpu_kernel void @vgpr42(ptr addrspace(1) %out, i32 %x) #0 {
  %h = call i32 @llvm.amdgcn.sched.vgpr.i32(i32 %x, i32 42, i32 1)
  %y = add i32 %h, 1
  store i32 %y, ptr addrspace(1) %out
  ret void
}

; --------------------------------------------------------------------------
; sched_vgpr: strict hint to v100

; GCN-LABEL: {{^}}vgpr100:
; GCN: v100
define amdgpu_kernel void @vgpr100(ptr addrspace(1) %out, i32 %x) #0 {
  %h = call i32 @llvm.amdgcn.sched.vgpr.i32(i32 %x, i32 100, i32 1)
  %y = add i32 %h, 1
  store i32 %y, ptr addrspace(1) %out
  ret void
}

; --------------------------------------------------------------------------
; sched_vgpr: strict hint to v0

; GCN-LABEL: {{^}}vgpr0:
; GCN: v0,
define amdgpu_kernel void @vgpr0(ptr addrspace(1) %out, i32 %x) #0 {
  %h = call i32 @llvm.amdgcn.sched.vgpr.i32(i32 %x, i32 0, i32 1)
  %y = add i32 %h, 1
  store i32 %y, ptr addrspace(1) %out
  ret void
}

; --------------------------------------------------------------------------
; sched_vgpr: strict hint to v512 (bank 2 territory)

; GCN-LABEL: {{^}}vgpr512:
; GCN: v{{(512|0 /\*v512\*/)}}

define amdgpu_kernel void @vgpr512(ptr addrspace(1) %out, i32 %x) #0 {
  %h = call i32 @llvm.amdgcn.sched.vgpr.i32(i32 %x, i32 512, i32 1)
  %y = add i32 %h, 1
  store i32 %y, ptr addrspace(1) %out
  ret void
}

; --------------------------------------------------------------------------
; sched_vgpr: 64-bit value, strict hint to v[200:201]

; GCN-LABEL: {{^}}vgpr200_64:
; GCN: v[200:201]
define amdgpu_kernel void @vgpr200_64(ptr addrspace(1) %out, i64 %x) #0 {
  %h = call i64 @llvm.amdgcn.sched.vgpr.i64(i64 %x, i32 200, i32 1)
  %y = add i64 %h, 1
  store i64 %y, ptr addrspace(1) %out
  ret void
}

; --------------------------------------------------------------------------
; sched_vgpr: 512-bit value (v16f32), strict hint to v[512:527]

; GCN-LABEL: {{^}}vgpr512_v16f32:
; GCN: v{{(\[512:513\]|\[0:1\] /\*v\[512:513\]\*/)}}

define amdgpu_kernel void @vgpr512_v16f32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %x = load <16 x float>, ptr addrspace(1) %in
  %h = call <16 x float> @llvm.amdgcn.sched.vgpr.v16f32(<16 x float> %x, i32 512, i32 1)
  store <16 x float> %h, ptr addrspace(1) %out
  ret void
}

; --------------------------------------------------------------------------
; Without hint: register allocator uses default low-numbered registers.
; Verify no high-numbered VGPRs appear.

; GCN-LABEL: {{^}}nohint:
; GCN-NOT: v42
; GCN-NOT: v100
; GCN-NOT: v512
; GCN: s_endpgm
define amdgpu_kernel void @nohint(ptr addrspace(1) %out, i32 %x) #0 {
  %y = add i32 %x, 1
  store i32 %y, ptr addrspace(1) %out
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="32,32" "amdgpu-waves-per-eu"="1,1" }

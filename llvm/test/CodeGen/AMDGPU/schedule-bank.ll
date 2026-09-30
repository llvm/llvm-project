; RUN: llc -mtriple=amdgcn -mcpu=gfx1250 < %s | FileCheck -check-prefix=GCN %s

; Verify that llvm.amdgcn.schedule.bank steers register allocation into the
; requested 256-register bank. The kernel is pinned to occupancy 1
; (waves-per-eu=1,1 with a single-wave workgroup) so the whole 1024-VGPR file is
; addressable and the high banks are actually available to the allocator.

declare i32 @llvm.amdgcn.schedule.bank.i32(i32, i32)

; A bank-2 hint places the value in v512-v767 and forces an s_set_vgpr_msb that
; latches bank 2 (src1=2) around its use.
; GCN-LABEL: {{^}}bank2:
; GCN: v_mov_b32_e32 v0 /*v512*/, s2
; GCN: s_set_vgpr_msb 0x8008
define amdgpu_kernel void @bank2(ptr addrspace(1) %out, i32 %x) #0 {
  %h = call i32 @llvm.amdgcn.schedule.bank.i32(i32 %x, i32 2)
  %y = add i32 %h, 1
  store i32 %y, ptr addrspace(1) %out
  ret void
}

; Without the hint the value stays in bank 0 and no bank switch is emitted.
; GCN-LABEL: {{^}}nohint:
; GCN-NOT: s_set_vgpr_msb
; GCN: s_endpgm
define amdgpu_kernel void @nohint(ptr addrspace(1) %out, i32 %x) #0 {
  %y = add i32 %x, 1
  store i32 %y, ptr addrspace(1) %out
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="32,32" "amdgpu-waves-per-eu"="1,1" }

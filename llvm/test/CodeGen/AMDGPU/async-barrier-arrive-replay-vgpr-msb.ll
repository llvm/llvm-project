; NOTE: Do not autogenerate. Check the ordered VGPR bank switch and contiguous EXEC-preserving expansion.
; RUN: llc -mtriple=amdgpu12.50-amd-amdhsa -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=amdgpu12.51-amd-amdhsa -verify-machineinstrs < %s | FileCheck %s

; Keep a divergent LDS address live across clobbers of all low VGPRs. The
; occupancy attributes allow allocation of v256 without spilling the address.
; The arrival uses the high bank, while the replay load needs v[0:1]. Switch
; banks before the expansion so the EXEC save, load and restore stay together.
define amdgpu_kernel void @high_vgpr_arrive() #0 {
; CHECK-LABEL: high_vgpr_arrive:
; CHECK: s_set_vgpr_msb 0x41
; CHECK: ds_atomic_async_barrier_arrive_b64 v0 /*v256*/
; CHECK-NEXT: s_wait_alu depctr_vm_vsrc(0)
; CHECK-NEXT: s_set_vgpr_msb 0x4100
; CHECK-NEXT: s_mov_b32 [[SAVED:s[0-9]+]], exec_lo
; CHECK-NEXT: s_mov_b32 exec_lo, 0
; CHECK-NEXT: global_load_async_to_lds_b8 v0, v[0:1], off
; CHECK-NEXT: s_mov_b32 exec_lo, [[SAVED]]
  %lane = call i32 @llvm.amdgcn.workitem.id.x()
  %address = shl i32 %lane, 3
  call void asm sideeffect "; clobber low VGPRs", "~{v[0:31]},~{v[32:63]},~{v[64:95]},~{v[96:127]},~{v[128:159]},~{v[160:191]},~{v[192:223]},~{v[224:255]}"()
  %barrier = inttoptr i32 %address to ptr addrspace(3)
  call void @llvm.amdgcn.ds.atomic.async.barrier.arrive.b64(ptr addrspace(3) %barrier)
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,32" "amdgpu-waves-per-eu"="1,1" }

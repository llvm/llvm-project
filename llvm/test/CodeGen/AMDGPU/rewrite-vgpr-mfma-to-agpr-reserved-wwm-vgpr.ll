; NOTE: Do not autogenerate. The checks target the scenario described below;
; regenerated full-function checks would hide its loss.
; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa < %s | FileCheck %s
; RUN: llc -enable-new-pm -mtriple=amdgpu9.50-amd-amdhsa < %s | FileCheck %s
; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -stop-before=amdgpu-rewrite-agpr-copy-mfma < %s | FileCheck --check-prefix=PRE %s
; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -stop-after=amdgpu-rewrite-agpr-copy-mfma < %s | FileCheck --check-prefix=POST %s

; The attributes leave v0..v31 and a0..a31. The clobbers force the SGPR spill
; lanes into v31, %mai through AGPRs and %vgpr into a spill slot where only v31
; looks free: lane 2 is still live, but v31 is reserved and LiveIntervals
; ignores uses of reserved registers. So neither the VGPR allocator nor
; AMDGPURewriteAGPRCopyMFMA's spill elimination may assign it.

; PRE and POST bracket AMDGPURewriteAGPRCopyMFMA: it must find the lane, a
; VGPR-form MFMA and the spill, and leave the MFMA rewritten and the spill in
; place. CHECK requires the lanes in v31, the spill around the second clobber,
; SGPR spills and exactly one VGPR spill, so no other spill can mask this one.
; PRE: $vgpr31 = SI_SPILL_S32_TO_VGPR
; PRE: V_MFMA_I32_4X4X4I8_vgprcd_e64
; PRE: SI_SPILL_AV32_SAVE
; POST: $vgpr31 = SI_SPILL_S32_TO_VGPR
; POST: V_MFMA_I32_4X4X4I8_e64
; POST: SI_SPILL_AV32_SAVE

; CHECK-LABEL: reserved_wwm_vgpr_not_in_alloc_order:
; CHECK: v_writelane_b32 v31,
; CHECK: scratch_store_dword off, v{{[0-9]+}}, off ; 4-byte Folded Spill
; CHECK: ; clobber vgprs and agprs
; CHECK: scratch_load_dword v{{[0-9]+}}, off, off ; 4-byte Folded Reload
; CHECK: .sgpr_spill_count: {{[1-9][0-9]*}}
; CHECK: .vgpr_spill_count: 1

define amdgpu_kernel void @reserved_wwm_vgpr_not_in_alloc_order(ptr addrspace(1) %ptr) #0 {
  %sgpr = call i32 asm sideeffect "; def $0", "=s"()
  call void asm sideeffect "; clobber sgprs", "~{s[0:31]},~{s33},~{s[34:35]},~{s[36:67]},~{s[68:71]},~{vcc}"()
  %acc = load <4 x i32>, ptr addrspace(1) %ptr
  %mai = call <4 x i32> @llvm.amdgcn.mfma.i32.4x4x4i8(i32 1, i32 2, <4 x i32> %acc, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  call void asm sideeffect "; clobber vgprs", "~{v[0:15]},~{v[16:23]},~{v[24:27]},~{v[28:29]},~{v30}"()
  store <4 x i32> %mai, ptr addrspace(1) %ptr
  %vgpr = call i32 asm sideeffect "; def $0", "=v"()
  call void asm sideeffect "; clobber vgprs and agprs", "~{v[0:15]},~{v[16:23]},~{v[24:27]},~{v[28:29]},~{v30},~{a[0:31]}"()
  call void asm sideeffect "; use $0", "v"(i32 %vgpr)
  call void asm sideeffect "; use $0", "s"(i32 %sgpr)
  ret void
}

declare <4 x i32> @llvm.amdgcn.mfma.i32.4x4x4i8(i32, i32, <4 x i32>, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.amdgcn.sched.barrier(i32 immarg)

attributes #0 = { "amdgpu-agpr-alloc"="32,32" "amdgpu-waves-per-eu"="8,8" }

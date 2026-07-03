; RUN: llc -mtriple=amdgcn -mcpu=gfx942 -stop-after=si-memory-legalizer < %s -o - | FileCheck %s --check-prefix=GFX942
; RUN: llc -mtriple=amdgcn -mcpu=gfx1200 -mattr=+wavefrontsize64 -stop-after=si-memory-legalizer < %s -o - | FileCheck %s --check-prefix=GFX12

define amdgpu_kernel void @workgroup_release_local_fence_single_wave() #0 {
; GFX942-LABEL: name: workgroup_release_local_fence_single_wave
; GFX942: bb.0 (%ir-block.0):
; GFX942-NEXT:   S_WAITCNT_soft .Lgkmcnt_0
; GFX942-NEXT:   S_WAITCNT_lds_direct
; GFX942-NEXT:   S_ENDPGM 0
;
; GFX12-LABEL: name: workgroup_release_local_fence_single_wave
; GFX12: bb.0 (%ir-block.0):
; GFX12-NEXT:   S_WAIT_DSCNT_soft 0
; GFX12-NEXT:   S_ENDPGM 0
  fence syncscope("workgroup") release, !mmra !{!"amdgpu-synchronize-as", !"local"}
  ret void
}

define amdgpu_kernel void @workgroup_release_global_local_fence_single_wave() #0 {
; GFX942-LABEL: name: workgroup_release_global_local_fence_single_wave
; GFX942: bb.0 (%ir-block.0):
; GFX942-NEXT:   S_WAITCNT_soft .Lgkmcnt_0
; GFX942-NEXT:   S_WAITCNT_lds_direct
; GFX942-NEXT:   S_ENDPGM 0
;
; GFX12-LABEL: name: workgroup_release_global_local_fence_single_wave
; GFX12: bb.0 (%ir-block.0):
; GFX12-NEXT:   S_WAIT_DSCNT_soft 0
; GFX12-NEXT:   S_ENDPGM 0
  fence syncscope("workgroup") release, !mmra !0
  ret void
}

define amdgpu_kernel void @workgroup_acquire_local_fence_single_wave() #0 {
; GFX942-LABEL: name: workgroup_acquire_local_fence_single_wave
; GFX942: bb.0 (%ir-block.0):
; GFX942-NEXT:   S_WAITCNT_soft .Lgkmcnt_0
; GFX942-NEXT:   S_ENDPGM 0
;
; GFX12-LABEL: name: workgroup_acquire_local_fence_single_wave
; GFX12: bb.0 (%ir-block.0):
; GFX12-NEXT:   S_WAIT_DSCNT_soft 0
; GFX12-NEXT:   S_ENDPGM 0
  fence syncscope("workgroup") acquire, !mmra !{!"amdgpu-synchronize-as", !"local"}
  ret void
}

define amdgpu_kernel void @workgroup_release_global_fence_single_wave() #0 {
; GFX942-LABEL: name: workgroup_release_global_fence_single_wave
; GFX942: bb.0 (%ir-block.0):
; GFX942-NEXT:   S_ENDPGM 0
;
; GFX12-LABEL: name: workgroup_release_global_fence_single_wave
; GFX12: bb.0 (%ir-block.0):
; GFX12-NEXT:   S_ENDPGM 0
  fence syncscope("workgroup") release, !mmra !{!"amdgpu-synchronize-as", !"global"}
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="64,64" nounwind }

!0 = !{!1, !2}
!1 = !{!"amdgpu-synchronize-as", !"global"}
!2 = !{!"amdgpu-synchronize-as", !"local"}

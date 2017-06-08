; RUN: not llc -march=amdgcn < %s 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: not llc -march=amdgcn < %s | FileCheck -check-prefix=GCN %s

; ERR: error: <unknown>:0:0: in function illegal_vgpr_to_sgpr_copy_i32 void (): illegal SGPR to VGPR copy
; GCN: ; illegal copy v1 to s9

define amdgpu_kernel void @illegal_vgpr_to_sgpr_copy_i32() #0 {
  %vgpr = call i32 asm sideeffect "; def $0", "=${v1}"()
  call void asm sideeffect "; use $0", "${s9}"(i32 %vgpr)
  ret void
}

; ERR: error: <unknown>:0:0: in function illegal_vgpr_to_sgpr_copy_v2i32 void (): illegal SGPR to VGPR copy
; GCN: ; illegal copy v[0:1] to s[10:11]
define amdgpu_kernel void @illegal_vgpr_to_sgpr_copy_v2i32() #0 {
  %vgpr = call <2 x i32> asm sideeffect "; def $0", "=${v[0:1]}"()
  call void asm sideeffect "; use $0", "${s[10:11]}"(<2 x i32> %vgpr)
  ret void
}

; ERR: error: <unknown>:0:0: in function illegal_vgpr_to_sgpr_copy_v4i32 void (): illegal SGPR to VGPR copy
; GCN: ; illegal copy v[0:3] to s[8:11]
define amdgpu_kernel void @illegal_vgpr_to_sgpr_copy_v4i32() #0 {
  %vgpr = call <4 x i32> asm sideeffect "; def $0", "=${v[0:3]}"()
  call void asm sideeffect "; use $0", "${s[8:11]}"(<4 x i32> %vgpr)
  ret void
}

; ERR: error: <unknown>:0:0: in function illegal_vgpr_to_sgpr_copy_v8i32 void (): illegal SGPR to VGPR copy
; GCN: ; illegal copy v[0:7] to s[8:15]
define amdgpu_kernel void @illegal_vgpr_to_sgpr_copy_v8i32() #0 {
  %vgpr = call <8 x i32> asm sideeffect "; def $0", "=${v[0:7]}"()
  call void asm sideeffect "; use $0", "${s[8:15]}"(<8 x i32> %vgpr)
  ret void
}

; ERR error: <unknown>:0:0: in function illegal_vgpr_to_sgpr_copy_v16i32 void (): illegal SGPR to VGPR copy
; GCN: ; illegal copy v[0:15] to s[16:31]
define amdgpu_kernel void @illegal_vgpr_to_sgpr_copy_v16i32() #0 {
  %vgpr = call <16 x i32> asm sideeffect "; def $0", "=${v[0:15]}"()
  call void asm sideeffect "; use $0", "${s[16:31]}"(<16 x i32> %vgpr)
  ret void
}

attributes #0 = { nounwind }

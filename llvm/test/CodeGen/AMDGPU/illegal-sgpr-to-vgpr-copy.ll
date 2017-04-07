; RUN: not llc -march=amdgcn < %s 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: not llc -march=amdgcn < %s | FileCheck -check-prefix=GCN %s

; ERR: error: <unknown>:0:0: in function illegal_vgpr_to_sgpr_copy_i32 void (): illegal SGPR to VGPR copy
; GCN: ; illegal copy v1 to s9

define amdgpu_kernel void @illegal_vgpr_to_sgpr_copy_i32() #0 {
  %vgpr = call i32 asm sideeffect "; def $0", "=${VGPR1}"()
  call void asm sideeffect "; use $0", "${SGPR9}"(i32 %vgpr)
  ret void
}

; ERR: error: <unknown>:0:0: in function illegal_vgpr_to_sgpr_copy_v2i32 void (): illegal SGPR to VGPR copy
; GCN: ; illegal copy v[0:1] to s[10:11]
define amdgpu_kernel void @illegal_vgpr_to_sgpr_copy_v2i32() #0 {
  %vgpr = call <2 x i32> asm sideeffect "; def $0", "=${VGPR0_VGPR1}"()
  call void asm sideeffect "; use $0", "${SGPR10_SGPR11}"(<2 x i32> %vgpr)
  ret void
}

; ERR: error: <unknown>:0:0: in function illegal_vgpr_to_sgpr_copy_v4i32 void (): illegal SGPR to VGPR copy
; GCN: ; illegal copy v[0:3] to s[8:11]
define amdgpu_kernel void @illegal_vgpr_to_sgpr_copy_v4i32() #0 {
  %vgpr = call <4 x i32> asm sideeffect "; def $0", "=${VGPR0_VGPR1_VGPR2_VGPR3}"()
  call void asm sideeffect "; use $0", "${SGPR8_SGPR9_SGPR10_SGPR11}"(<4 x i32> %vgpr)
  ret void
}

; ERR: error: <unknown>:0:0: in function illegal_vgpr_to_sgpr_copy_v8i32 void (): illegal SGPR to VGPR copy
; GCN: ; illegal copy v[0:7] to s[8:15]
define amdgpu_kernel void @illegal_vgpr_to_sgpr_copy_v8i32() #0 {
  %vgpr = call <8 x i32> asm sideeffect "; def $0", "=${VGPR0_VGPR1_VGPR2_VGPR3_VGPR4_VGPR5_VGPR6_VGPR7}"()
  call void asm sideeffect "; use $0", "${SGPR8_SGPR9_SGPR10_SGPR11_SGPR12_SGPR13_SGPR14_SGPR15}"(<8 x i32> %vgpr)
  ret void
}

; ERR error: <unknown>:0:0: in function illegal_vgpr_to_sgpr_copy_v16i32 void (): illegal SGPR to VGPR copy
; GCN: ; illegal copy v[0:15] to s[16:31]
define amdgpu_kernel void @illegal_vgpr_to_sgpr_copy_v16i32() #0 {
  %vgpr = call <16 x i32> asm sideeffect "; def $0", "=${VGPR0_VGPR1_VGPR2_VGPR3_VGPR4_VGPR5_VGPR6_VGPR7_VGPR8_VGPR9_VGPR10_VGPR11_VGPR12_VGPR13_VGPR14_VGPR15}"()
  call void asm sideeffect "; use $0", "${SGPR16_SGPR17_SGPR18_SGPR19_SGPR20_SGPR21_SGPR22_SGPR23_SGPR24_SGPR25_SGPR26_SGPR27_SGPR28_SGPR29_SGPR30_SGPR31}"(<16 x i32> %vgpr)
  ret void
}

attributes #0 = { nounwind }

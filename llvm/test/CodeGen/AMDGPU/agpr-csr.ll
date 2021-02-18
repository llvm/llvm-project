; RUN: llc -march=amdgcn -mcpu=gfx90a -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX90A %s

; GCN-LABEL: {{^}}func_empty:
; GCN-NOT: buffer_
; GCN-NOT: v_accvgpr
; GCN:     s_setpc_b64
define void @func_empty() #0 {
  ret void
}

; GCN-LABEL: {{^}}func_areg_4:
; GCN-NOT: buffer_
; GCN-NOT: v_accvgpr
; GCN: use agpr3
; GCN-NOT: buffer_
; GCN-NOT: v_accvgpr
; GCN: s_setpc_b64
define void @func_areg_4() #0 {
  call void asm sideeffect "; use agpr3", "~{a3}" ()
  ret void
}

; GCN-LABEL: {{^}}func_areg_32:
; GCN-NOT: buffer_
; GCN-NOT: v_accvgpr
; GCN: use agpr31
; GCN-NOT: buffer_
; GCN-NOT: v_accvgpr
; GCN: s_setpc_b64
define void @func_areg_32() #0 {
  call void asm sideeffect "; use agpr31", "~{a31}" ()
  ret void
}

; GCN-LABEL: {{^}}func_areg_33:
; GFX908-NOT: buffer_
; GCN-NOT:    v_accvgpr
; GCN:        use agpr32
; GFX908-NOT: buffer_
; GCN-NOT:    v_accvgpr
; GCN:        s_setpc_b64
define void @func_areg_33() #0 {
  call void asm sideeffect "; use agpr32", "~{a32}" ()
  ret void
}

; GCN-LABEL: {{^}}func_areg_64:
; GFX908-NOT: buffer_
; GCN-NOT:    v_accvgpr
; GFX90A:     buffer_store_dword a63,
; GCN:        use agpr63
; GFX90A:     buffer_load_dword a63,
; GCN-NOT:    v_accvgpr
; GCN:        s_setpc_b64
define void @func_areg_64() #0 {
  call void asm sideeffect "; use agpr63", "~{a63}" ()
  ret void
}

; GCN-LABEL: {{^}}func_areg_31_63:
; GFX908-NOT: buffer_
; GCN-NOT:    v_accvgpr
; GFX90A:     buffer_store_dword a63,
; GCN:        use agpr31, agpr63
; GFX90A:     buffer_load_dword a63,
; GCN-NOT:    buffer_
; GCN-NOT:    v_accvgpr
; GCN:        s_setpc_b64
define void @func_areg_31_63() #0 {
  call void asm sideeffect "; use agpr31, agpr63", "~{a31},~{a63}" ()
  ret void
}

declare void @func_unknown() #0

; GCN-LABEL: {{^}}test_call_empty:
; GCN-NOT:         buffer_
; GCN-NOT:         v_accvgpr
; GCN:             def a[0:31]
; GFX908-COUNT-8:  v_accvgpr_read_b32
; GFX90A-NOT:      v_accvgpr
; GCN-NOT:         buffer_
; GCN:             s_swappc_b64
; GCN-NOT:         buffer_
; GFX90A-NOT:      v_accvgpr
; GFX908-COUNT-8:  global_store_dwordx4 v[{{[0-9:]+}}], v[{{[0-9:]+}}]
; GFX90A-COUNT-8:  global_store_dwordx4 v[{{[0-9:]+}}], a[{{[0-9:]+}}]
; GCN:             s_endpgm
define amdgpu_kernel void @test_call_empty() #0 {
bb:
  %reg = call <32 x float> asm sideeffect "; def $0", "=a"()
  call void @func_empty()
  store volatile <32 x float> %reg, <32 x float> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}test_call_areg4:
; GCN-NOT:         buffer_
; GCN-NOT:         v_accvgpr
; GFX908:          def a[0:31]
; GFX90A:          def a[4:35]
; GFX908-COUNT-8:  v_accvgpr_read_b32
; GFX90A-NOT:      v_accvgpr
; GCN-NOT:         buffer_
; GCN:             s_swappc_b64
; GCN-NOT:         buffer_
; GFX90A-NOT:      v_accvgpr
; GFX908-COUNT-8:  global_store_dwordx4 v[{{[0-9:]+}}], v[{{[0-9:]+}}]
; GFX90A-COUNT-8:  global_store_dwordx4 v[{{[0-9:]+}}], a[{{[0-9:]+}}]
; GCN:             s_endpgm
define amdgpu_kernel void @test_call_areg4() #0 {
bb:
  %reg = call <32 x float> asm sideeffect "; def $0", "=a"()
  call void @func_areg_4()
  store volatile <32 x float> %reg, <32 x float> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}test_call_areg32:
; GCN-NOT:         buffer_
; GCN-NOT:         v_accvgpr
; GFX908:          def a[0:31]
; GFX90A:          def a[32:63]
; GFX908-COUNT-8:  v_accvgpr_read_b32
; GFX90A-NOT:      v_accvgpr
; GCN-NOT:         buffer_
; GCN:             s_swappc_b64
; GCN-NOT:         buffer_
; GFX90A-NOT:      v_accvgpr
; GFX908-COUNT-8:  global_store_dwordx4 v[{{[0-9:]+}}], v[{{[0-9:]+}}]
; GFX90A-COUNT-8:  global_store_dwordx4 v[{{[0-9:]+}}], a[{{[0-9:]+}}]
; GCN:             s_endpgm
define amdgpu_kernel void @test_call_areg32() #0 {
bb:
  %reg = call <32 x float> asm sideeffect "; def $0", "=a"()
  call void @func_areg_32()
  store volatile <32 x float> %reg, <32 x float> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}test_call_areg64:
; GCN-NOT:         buffer_
; GCN-NOT:         v_accvgpr
; GCN:             def a[0:31]
; GFX908-COUNT-8:  v_accvgpr_read_b32
; GFX90A-NOT:      v_accvgpr
; GCN-NOT:         buffer_
; GCN:             s_swappc_b64
; GCN-NOT:         buffer_
; GFX90A-NOT:      v_accvgpr
; GFX908-COUNT-8:  global_store_dwordx4 v[{{[0-9:]+}}], v[{{[0-9:]+}}]
; GFX90A-COUNT-8:  global_store_dwordx4 v[{{[0-9:]+}}], a[{{[0-9:]+}}]
; GCN:             s_endpgm
define amdgpu_kernel void @test_call_areg64() #0 {
bb:
  %reg = call <32 x float> asm sideeffect "; def $0", "=a"()
  call void @func_areg_64()
  store volatile <32 x float> %reg, <32 x float> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}test_call_areg31_63:
; GCN-NOT:         buffer_
; GCN-NOT:         v_accvgpr
; GFX908:          def a[0:31]
; GFX90A:          def a[32:63]
; GFX908-COUNT-8:  v_accvgpr_read_b32
; GFX90A-NOT:      v_accvgpr
; GCN-NOT:         buffer_
; GCN:             s_swappc_b64
; GCN-NOT:         buffer_
; GFX90A-NOT:      v_accvgpr
; GFX908-COUNT-8:  global_store_dwordx4 v[{{[0-9:]+}}], v[{{[0-9:]+}}]
; GFX90A-COUNT-8:  global_store_dwordx4 v[{{[0-9:]+}}], a[{{[0-9:]+}}]
; GCN:             s_endpgm
define amdgpu_kernel void @test_call_areg31_63() #0 {
bb:
  %reg = call <32 x float> asm sideeffect "; def $0", "=a"()
  call void @func_areg_31_63()
  store volatile <32 x float> %reg, <32 x float> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}test_call_unknown:
; GCN-NOT:         buffer_
; GCN-NOT:         v_accvgpr
; GFX908:          def a[0:31]
; GFX90A:          def a[32:63]
; GFX908-COUNT-8:  v_accvgpr_read_b32
; GFX90A-NOT:      v_accvgpr
; GCN-NOT:         buffer_
; GCN:             s_swappc_b64
; GCN-NOT:         buffer_
; GFX90A-NOT:      v_accvgpr
; GFX908-COUNT-8:  global_store_dwordx4 v[{{[0-9:]+}}], v[{{[0-9:]+}}]
; GFX90A-COUNT-8:  global_store_dwordx4 v[{{[0-9:]+}}], a[{{[0-9:]+}}]
; GCN:             s_endpgm
define amdgpu_kernel void @test_call_unknown() #0 {
bb:
  %reg = call <32 x float> asm sideeffect "; def $0", "=a"()
  call void @func_unknown()
  store volatile <32 x float> %reg, <32 x float> addrspace(1)* undef
  ret void
}

attributes #0 = { nounwind noinline "amdgpu-flat-work-group-size"="1,512" }

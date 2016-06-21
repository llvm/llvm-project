; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: {{^}}inline_asm:
; CHECK: s_endpgm
; CHECK: s_endpgm
define void @inline_asm(i32 addrspace(1)* %out) {
entry:
  store i32 5, i32 addrspace(1)* %out
  call void asm sideeffect "s_endpgm", ""()
  ret void
}

; CHECK-LABEL: {{^}}inline_asm_shader:
; CHECK: s_endpgm
; CHECK: s_endpgm
define amdgpu_ps void @inline_asm_shader() {
entry:
  call void asm sideeffect "s_endpgm", ""()
  ret void
}


; CHECK: {{^}}branch_on_asm:
; Make sure inline assembly is treted as divergent.
; CHECK: s_mov_b32 s{{[0-9]+}}, 0
; CHECK: s_and_saveexec_b64
define void @branch_on_asm(i32 addrspace(1)* %out) {
	%zero = call i32 asm "s_mov_b32 $0, 0", "=s"()
	%cmp = icmp eq i32 %zero, 0
	br i1 %cmp, label %if, label %endif

if:
	store i32 0, i32 addrspace(1)* %out
	br label %endif

endif:
  ret void
}

; CHECK-LABEL: {{^}}v_cmp_asm:
; CHECK: v_mov_b32_e32 [[SRC:v[0-9]+]], s{{[0-9]+}}
; CHECK: v_cmp_ne_i32_e64 s{{\[}}[[MASK_LO:[0-9]+]]:[[MASK_HI:[0-9]+]]{{\]}}, 0, [[SRC]]
; CHECK-DAG: v_mov_b32_e32 v[[V_LO:[0-9]+]], s[[MASK_LO]]
; CHECK-DAG: v_mov_b32_e32 v[[V_HI:[0-9]+]], s[[MASK_HI]]
; CHECK: buffer_store_dwordx2 v{{\[}}[[V_LO]]:[[V_HI]]{{\]}}
define void @v_cmp_asm(i64 addrspace(1)* %out, i32 %in) {
  %sgpr = tail call i64 asm "v_cmp_ne_i32_e64 $0, 0, $1", "=s,v"(i32 %in)
  store i64 %sgpr, i64 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}code_size_inline_asm:
; CHECK: codeLenInByte = 12
define void @code_size_inline_asm(i32 addrspace(1)* %out) {
entry:
  call void asm sideeffect "v_nop_e64", ""()
  ret void
}

; All inlineasm instructions are assumed to be the maximum size
; CHECK-LABEL: {{^}}code_size_inline_asm_small_inst:
; CHECK: codeLenInByte = 12
define void @code_size_inline_asm_small_inst(i32 addrspace(1)* %out) {
entry:
  call void asm sideeffect "v_nop_e32", ""()
  ret void
}

; CHECK-LABEL: {{^}}code_size_inline_asm_2_inst:
; CHECK: codeLenInByte = 20
define void @code_size_inline_asm_2_inst(i32 addrspace(1)* %out) {
entry:
  call void asm sideeffect "
    v_nop_e64
    v_nop_e64
   ", ""()
  ret void
}

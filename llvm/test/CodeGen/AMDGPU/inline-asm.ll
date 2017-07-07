; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: {{^}}inline_asm:
; CHECK: s_endpgm
; CHECK: s_endpgm
define amdgpu_kernel void @inline_asm(i32 addrspace(1)* %out) {
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
define amdgpu_kernel void @branch_on_asm(i32 addrspace(1)* %out) {
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
; CHECK: v_cmp_ne_u32_e64 s{{\[}}[[MASK_LO:[0-9]+]]:[[MASK_HI:[0-9]+]]{{\]}}, 0, [[SRC]]
; CHECK-DAG: v_mov_b32_e32 v[[V_LO:[0-9]+]], s[[MASK_LO]]
; CHECK-DAG: v_mov_b32_e32 v[[V_HI:[0-9]+]], s[[MASK_HI]]
; CHECK: buffer_store_dwordx2 v{{\[}}[[V_LO]]:[[V_HI]]{{\]}}
define amdgpu_kernel void @v_cmp_asm(i64 addrspace(1)* %out, i32 %in) {
  %sgpr = tail call i64 asm "v_cmp_ne_u32_e64 $0, 0, $1", "=s,v"(i32 %in)
  store i64 %sgpr, i64 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}code_size_inline_asm:
; CHECK: codeLenInByte = 12
define amdgpu_kernel void @code_size_inline_asm(i32 addrspace(1)* %out) {
entry:
  call void asm sideeffect "v_nop_e64", ""()
  ret void
}

; All inlineasm instructions are assumed to be the maximum size
; CHECK-LABEL: {{^}}code_size_inline_asm_small_inst:
; CHECK: codeLenInByte = 12
define amdgpu_kernel void @code_size_inline_asm_small_inst(i32 addrspace(1)* %out) {
entry:
  call void asm sideeffect "v_nop_e32", ""()
  ret void
}

; CHECK-LABEL: {{^}}code_size_inline_asm_2_inst:
; CHECK: codeLenInByte = 20
define amdgpu_kernel void @code_size_inline_asm_2_inst(i32 addrspace(1)* %out) {
entry:
  call void asm sideeffect "
    v_nop_e64
    v_nop_e64
   ", ""()
  ret void
}

; CHECK-LABEL: {{^}}code_size_inline_asm_2_inst_extra_newline:
; CHECK: codeLenInByte = 20
define amdgpu_kernel void @code_size_inline_asm_2_inst_extra_newline(i32 addrspace(1)* %out) {
entry:
  call void asm sideeffect "
    v_nop_e64

    v_nop_e64
   ", ""()
  ret void
}

; CHECK-LABEL: {{^}}code_size_inline_asm_0_inst:
; CHECK: codeLenInByte = 4
define amdgpu_kernel void @code_size_inline_asm_0_inst(i32 addrspace(1)* %out) {
entry:
  call void asm sideeffect "", ""()
  ret void
}

; CHECK-LABEL: {{^}}code_size_inline_asm_1_comment:
; CHECK: codeLenInByte = 4
define amdgpu_kernel void @code_size_inline_asm_1_comment(i32 addrspace(1)* %out) {
entry:
  call void asm sideeffect "; comment", ""()
  ret void
}

; CHECK-LABEL: {{^}}code_size_inline_asm_newline_1_comment:
; CHECK: codeLenInByte = 4
define amdgpu_kernel void @code_size_inline_asm_newline_1_comment(i32 addrspace(1)* %out) {
entry:
  call void asm sideeffect "
; comment", ""()
  ret void
}

; CHECK-LABEL: {{^}}code_size_inline_asm_1_comment_newline:
; CHECK: codeLenInByte = 4
define amdgpu_kernel void @code_size_inline_asm_1_comment_newline(i32 addrspace(1)* %out) {
entry:
  call void asm sideeffect "; comment
", ""()
  ret void
}

; CHECK-LABEL: {{^}}code_size_inline_asm_2_comments_line:
; CHECK: codeLenInByte = 4
define amdgpu_kernel void @code_size_inline_asm_2_comments_line(i32 addrspace(1)* %out) {
entry:
  call void asm sideeffect "; first comment ; second comment", ""()
  ret void
}

; CHECK-LABEL: {{^}}code_size_inline_asm_2_comments_line_nospace:
; CHECK: codeLenInByte = 4
define amdgpu_kernel void @code_size_inline_asm_2_comments_line_nospace(i32 addrspace(1)* %out) {
entry:
  call void asm sideeffect "; first comment;second comment", ""()
  ret void
}

; CHECK-LABEL: {{^}}code_size_inline_asm_mixed_comments0:
; CHECK: codeLenInByte = 20
define amdgpu_kernel void @code_size_inline_asm_mixed_comments0(i32 addrspace(1)* %out) {
entry:
  call void asm sideeffect "; comment
    v_nop_e64 ; inline comment
; separate comment
    v_nop_e64

    ; trailing comment
    ; extra comment
  ", ""()
  ret void
}

; CHECK-LABEL: {{^}}code_size_inline_asm_mixed_comments1:
; CHECK: codeLenInByte = 20
define amdgpu_kernel void @code_size_inline_asm_mixed_comments1(i32 addrspace(1)* %out) {
entry:
  call void asm sideeffect "v_nop_e64 ; inline comment
; separate comment
    v_nop_e64

    ; trailing comment
    ; extra comment
  ", ""()
  ret void
}

; CHECK-LABEL: {{^}}code_size_inline_asm_mixed_comments_operands:
; CHECK: codeLenInByte = 20
define amdgpu_kernel void @code_size_inline_asm_mixed_comments_operands(i32 addrspace(1)* %out) {
entry:
  call void asm sideeffect "; comment
    v_add_i32_e32 v0, vcc, v1, v2 ; inline comment
; separate comment
    v_bfrev_b32_e32 v0, 1

    ; trailing comment
    ; extra comment
  ", ""()
  ret void
}

; FIXME: Should not have intermediate sgprs
; CHECK-LABEL: {{^}}i64_imm_input_phys_vgpr:
; CHECK: s_mov_b32 s1, 0
; CHECK: s_mov_b32 s0, 0x1e240
; CHECK: v_mov_b32_e32 v0, s0
; CHECK: v_mov_b32_e32 v1, s1
; CHECK: use v[0:1]
define amdgpu_kernel void @i64_imm_input_phys_vgpr() {
entry:
  call void asm sideeffect "; use $0 ", "{v[0:1]}"(i64 123456)
  ret void
}

; CHECK-LABEL: {{^}}i1_imm_input_phys_vgpr:
; CHECK: v_mov_b32_e32 v0, -1{{$}}
; CHECK: ; use v0
define amdgpu_kernel void @i1_imm_input_phys_vgpr() {
entry:
  call void asm sideeffect "; use $0 ", "{v0}"(i1 true)
  ret void
}

; CHECK-LABEL: {{^}}i1_input_phys_vgpr:
; CHECK: {{buffer|flat}}_load_ubyte [[LOAD:v[0-9]+]]
; CHECK: v_and_b32_e32 [[LOAD]], 1, [[LOAD]]
; CHECK-NEXT: v_cmp_eq_u32_e32 vcc, 1, [[LOAD]]
; CHECK-NEXT: v_cndmask_b32_e64 v0, 0, -1, vcc
; CHECK: ; use v0
define amdgpu_kernel void @i1_input_phys_vgpr() {
entry:
  %val = load i1, i1 addrspace(1)* undef
  call void asm sideeffect "; use $0 ", "{v0}"(i1 %val)
  ret void
}

; FIXME: Should be scheduled to shrink vcc
; CHECK-LABEL: {{^}}i1_input_phys_vgpr_x2:
; CHECK: v_cmp_eq_u32_e32 vcc, 1, v0
; CHECK: v_cndmask_b32_e64 v0, 0, -1, vcc
; CHECK: v_cmp_eq_u32_e32 vcc, 1, v1
; CHECK: v_cndmask_b32_e64 v1, 0, -1, vcc
define amdgpu_kernel void @i1_input_phys_vgpr_x2() {
entry:
  %val0 = load volatile i1, i1 addrspace(1)* undef
  %val1 = load volatile i1, i1 addrspace(1)* undef
  call void asm sideeffect "; use $0 $1 ", "{v0}, {v1}"(i1 %val0, i1 %val1)
  ret void
}

; CHECK-LABEL: {{^}}muliple_def_phys_vgpr:
; CHECK: ; def v0
; CHECK: v_mov_b32_e32 v1, v0
; CHECK: ; def v0
; CHECK: v_lshlrev_b32_e32 v{{[0-9]+}}, v0, v1
define amdgpu_kernel void @muliple_def_phys_vgpr() {
entry:
  %def0 = call i32 asm sideeffect "; def $0 ", "={v0}"()
  %def1 = call i32 asm sideeffect "; def $0 ", "={v0}"()
  %add = shl i32 %def0, %def1
  store i32 %add, i32 addrspace(1)* undef
  ret void
}

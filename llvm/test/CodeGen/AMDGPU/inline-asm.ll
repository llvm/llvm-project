; RUN: llc -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck --check-prefix=CHECK %s
; RUN: llc -mtriple=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck  --check-prefix=CHECK %s

; CHECK-LABEL: {{^}}inline_asm:
; CHECK: s_endpgm
; CHECK: s_endpgm
define amdgpu_kernel void @inline_asm(ptr addrspace(1) %out) {
entry:
  store i32 5, ptr addrspace(1) %out
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


; CHECK-LABEL: {{^}}branch_on_asm_vgpr:
; Make sure VGPR inline assembly is treated as divergent.
; CHECK: v_mov_b32 v{{[0-9]+}}, 0
; CHECK: v_cmp_eq_u32
; CHECK: s_and_saveexec_b64
define amdgpu_kernel void @branch_on_asm_vgpr(ptr addrspace(1) %out) {
	%zero = call i32 asm "v_mov_b32 $0, 0", "=v"()
	%cmp = icmp eq i32 %zero, 0
	br i1 %cmp, label %if, label %endif

if:
	store i32 0, ptr addrspace(1) %out
	br label %endif

endif:
  ret void
}

; CHECK-LABEL: {{^}}branch_on_asm_sgpr:
; Make sure SGPR inline assembly is treated as uniform
; CHECK: s_mov_b32 s{{[0-9]+}}, 0
; CHECK: s_cmp_lg_u32
; CHECK: s_cbranch_scc0
define amdgpu_kernel void @branch_on_asm_sgpr(ptr addrspace(1) %out) {
	%zero = call i32 asm "s_mov_b32 $0, 0", "=s"()
	%cmp = icmp eq i32 %zero, 0
	br i1 %cmp, label %if, label %endif

if:
	store i32 0, ptr addrspace(1) %out
	br label %endif

endif:
  ret void
}

; CHECK-LABEL: {{^}}v_cmp_asm:
; CHECK: v_mov_b32_e32 [[SRC:v[0-9]+]], s{{[0-9]+}}
; CHECK: v_cmp_ne_u32_e64 s[[[MASK_LO:[0-9]+]]:[[MASK_HI:[0-9]+]]], 0, [[SRC]]
; CHECK-DAG: v_mov_b32_e32 v[[V_LO:[0-9]+]], s[[MASK_LO]]
; CHECK-DAG: v_mov_b32_e32 v[[V_HI:[0-9]+]], s[[MASK_HI]]
; CHECK: buffer_store_dwordx2 v[[[V_LO]]:[[V_HI]]]
define amdgpu_kernel void @v_cmp_asm(ptr addrspace(1) %out, i32 %in) {
  %sgpr = tail call i64 asm "v_cmp_ne_u32_e64 $0, 0, $1", "=s,v"(i32 %in)
  store i64 %sgpr, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}code_size_inline_asm:
; CHECK: codeLenInByte = 12
define amdgpu_kernel void @code_size_inline_asm(ptr addrspace(1) %out) {
entry:
  call void asm sideeffect "v_nop_e64", ""()
  ret void
}

; All inlineasm instructions are assumed to be the maximum size
; CHECK-LABEL: {{^}}code_size_inline_asm_small_inst:
; CHECK: codeLenInByte = 12
define amdgpu_kernel void @code_size_inline_asm_small_inst(ptr addrspace(1) %out) {
entry:
  call void asm sideeffect "v_nop_e32", ""()
  ret void
}

; CHECK-LABEL: {{^}}code_size_inline_asm_2_inst:
; CHECK: codeLenInByte = 20
define amdgpu_kernel void @code_size_inline_asm_2_inst(ptr addrspace(1) %out) {
entry:
  call void asm sideeffect "
    v_nop_e64
    v_nop_e64
   ", ""()
  ret void
}

; CHECK-LABEL: {{^}}code_size_inline_asm_2_inst_extra_newline:
; CHECK: codeLenInByte = 20
define amdgpu_kernel void @code_size_inline_asm_2_inst_extra_newline(ptr addrspace(1) %out) {
entry:
  call void asm sideeffect "
    v_nop_e64

    v_nop_e64
   ", ""()
  ret void
}

; CHECK-LABEL: {{^}}code_size_inline_asm_0_inst:
; CHECK: codeLenInByte = 4
define amdgpu_kernel void @code_size_inline_asm_0_inst(ptr addrspace(1) %out) {
entry:
  call void asm sideeffect "", ""()
  ret void
}

; CHECK-LABEL: {{^}}code_size_inline_asm_1_comment:
; CHECK: codeLenInByte = 4
define amdgpu_kernel void @code_size_inline_asm_1_comment(ptr addrspace(1) %out) {
entry:
  call void asm sideeffect "; comment", ""()
  ret void
}

; CHECK-LABEL: {{^}}code_size_inline_asm_newline_1_comment:
; CHECK: codeLenInByte = 4
define amdgpu_kernel void @code_size_inline_asm_newline_1_comment(ptr addrspace(1) %out) {
entry:
  call void asm sideeffect "
; comment", ""()
  ret void
}

; CHECK-LABEL: {{^}}code_size_inline_asm_1_comment_newline:
; CHECK: codeLenInByte = 4
define amdgpu_kernel void @code_size_inline_asm_1_comment_newline(ptr addrspace(1) %out) {
entry:
  call void asm sideeffect "; comment
", ""()
  ret void
}

; CHECK-LABEL: {{^}}code_size_inline_asm_2_comments_line:
; CHECK: codeLenInByte = 4
define amdgpu_kernel void @code_size_inline_asm_2_comments_line(ptr addrspace(1) %out) {
entry:
  call void asm sideeffect "; first comment ; second comment", ""()
  ret void
}

; CHECK-LABEL: {{^}}code_size_inline_asm_2_comments_line_nospace:
; CHECK: codeLenInByte = 4
define amdgpu_kernel void @code_size_inline_asm_2_comments_line_nospace(ptr addrspace(1) %out) {
entry:
  call void asm sideeffect "; first comment;second comment", ""()
  ret void
}

; CHECK-LABEL: {{^}}code_size_inline_asm_mixed_comments0:
; CHECK: codeLenInByte = 20
define amdgpu_kernel void @code_size_inline_asm_mixed_comments0(ptr addrspace(1) %out) {
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
define amdgpu_kernel void @code_size_inline_asm_mixed_comments1(ptr addrspace(1) %out) {
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
define amdgpu_kernel void @code_size_inline_asm_mixed_comments_operands(ptr addrspace(1) %out) {
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
; CHECK: v_mov_b32_e32 v0, 0x1e240
; CHECK: v_mov_b32_e32 v1, 0
; CHECK: use v[0:1]
define amdgpu_kernel void @i64_imm_input_phys_vgpr() {
entry:
  call void asm sideeffect "; use $0 ", "{v[0:1]}"(i64 123456)
  ret void
}

; CHECK-LABEL: {{^}}i1_imm_input_phys_vgpr:
; CHECK: v_mov_b32_e32 v0, 1{{$}}
; CHECK: ; use v0
define amdgpu_kernel void @i1_imm_input_phys_vgpr() {
entry:
  call void asm sideeffect "; use $0 ", "{v0}"(i1 true)
  ret void
}


; FIXME: This behavior is nonsense. We should probably disallow i1 asm

; CHECK-LABEL: {{^}}i1_input_phys_vgpr:
; CHECK: {{buffer|flat}}_load_ubyte [[LOAD:v[0-9]+]]
; CHECK-NOT: [[LOAD]]
; CHECK: ; use v0
; CHECK: v_and_b32_e32 [[STORE:v[0-9]+]], 1, v1
; CHECK: {{buffer|flat}}_store_byte [[STORE]],
define amdgpu_kernel void @i1_input_phys_vgpr() {
entry:
  %val = load i1, ptr addrspace(1) undef
  %cc = call i1 asm sideeffect "; use $1, def $0 ", "={v1}, {v0}"(i1 %val)
  store i1 %cc, ptr addrspace(1) undef
  ret void
}

; FIXME: Should prodbably be masking high bits of load.
; CHECK-LABEL: {{^}}i1_input_phys_vgpr_x2:
; CHECK: buffer_load_ubyte v0
; CHECK-NEXT: s_waitcnt
; CHECK-NEXT: buffer_load_ubyte v1
; CHECK-NEXT: s_waitcnt
; CHECK-NEXT: ASMSTART
define amdgpu_kernel void @i1_input_phys_vgpr_x2() {
entry:
  %val0 = load volatile i1, ptr addrspace(1) undef
  %val1 = load volatile i1, ptr addrspace(1) undef
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
  store i32 %add, ptr addrspace(1) undef
  ret void
}

; CHECK-LABEL: {{^}}asm_constraint_c_n:
; CHECK: s_trap 10{{$}}
define amdgpu_kernel void @asm_constraint_c_n()  {
entry:
  tail call void asm sideeffect "s_trap ${0:c}", "n"(i32 10) #1
  ret void
}

; CHECK-LABEL: {{^}}asm_constraint_n_n:
; CHECK: s_trap -10{{$}}
define amdgpu_kernel void @asm_constraint_n_n()  {
entry:
  tail call void asm sideeffect "s_trap ${0:n}", "n"(i32 10) #1
  ret void
}

; Make sure tuples of 3 SGPRs are printed with the [] syntax instead
; of the tablegen default.
; CHECK-LABEL: {{^}}sgpr96_name_format:
; CHECK: ; sgpr96 s[0:2]
define amdgpu_kernel void @sgpr96_name_format()  {
entry:
  tail call void asm sideeffect "; sgpr96 $0", "s"(<3 x i32> <i32 10, i32 11, i32 12>) #1
  ret void
}

; Check aggregate types are handled properly.
; CHECK-LABEL: mad_u64
; CHECK: v_mad_u64_u32
define void @mad_u64(i32 %x, i1 %c0) {
entry:
  br i1 %c0, label %exit, label %false

false:
  %s0 = tail call { i64, i64 } asm sideeffect "v_mad_u64_u32 $0, $1, $2, $3, $4", "=v,=s,v,v,v"(i32 -766435501, i32 %x, i64 0)
  br label %exit

exit:
  %s1 = phi { i64, i64} [ undef, %entry ], [ %s0, %false]
  %v0 = extractvalue { i64, i64 } %s1, 0
  %v1 = extractvalue { i64, i64 } %s1, 1
  tail call void asm sideeffect "; use $0", "v"(i64 %v0)
  tail call void asm sideeffect "; use $0", "v"(i64 %v1)
  ret void
}

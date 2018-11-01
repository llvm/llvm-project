; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s
;
;
; Most SALU instructions ignore control flow, so we need to make sure
; they don't overwrite values from other blocks.

; If the branch decision is made based on a value in an SGPR then all
; threads will execute the same code paths, so we don't need to worry
; about instructions in different blocks overwriting each other.
; SI-LABEL: {{^}}sgpr_if_else_salu_br:
; SI: s_add
; SI: s_branch

; SI: s_sub

define amdgpu_kernel void @sgpr_if_else_salu_br(i32 addrspace(1)* %out, i32 %a, i32 %b, i32 %c, i32 %d, i32 %e) {
entry:
  %0 = icmp eq i32 %a, 0
  br i1 %0, label %if, label %else

if:
  %1 = sub i32 %b, %c
  br label %endif

else:
  %2 = add i32 %d, %e
  br label %endif

endif:
  %3 = phi i32 [%1, %if], [%2, %else]
  %4 = add i32 %3, %a
  store i32 %4, i32 addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}sgpr_if_else_salu_br_opt:
; SI: s_cmp_lg_u32
; SI: s_cbranch_scc0 [[IF:BB[0-9]+_[0-9]+]]

; SI: ; %bb.1: ; %else
; SI: s_load_dword [[LOAD0:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x2e
; SI: s_load_dword [[LOAD1:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x37
; SI-NOT: add
; SI: s_branch [[ENDIF:BB[0-9]+_[0-9]+]]

; SI: [[IF]]: ; %if
; SI: s_load_dword [[LOAD0]], s{{\[[0-9]+:[0-9]+\]}}, 0x1c
; SI: s_load_dword [[LOAD1]], s{{\[[0-9]+:[0-9]+\]}}, 0x25
; SI-NOT: add

; SI: [[ENDIF]]: ; %endif
; SI: s_add_i32 s{{[0-9]+}}, [[LOAD0]], [[LOAD1]]
; SI: buffer_store_dword
; SI-NEXT: s_endpgm
define amdgpu_kernel void @sgpr_if_else_salu_br_opt(i32 addrspace(1)* %out, [8 x i32], i32 %a, [8 x i32], i32 %b, [8 x i32], i32 %c, [8 x i32], i32 %d, [8 x i32], i32 %e) {
entry:
  %cmp0 = icmp eq i32 %a, 0
  br i1 %cmp0, label %if, label %else

if:
  %add0 = add i32 %b, %c
  br label %endif

else:
  %add1 = add i32 %d, %e
  br label %endif

endif:
  %phi = phi i32 [%add0, %if], [%add1, %else]
  %add2 = add i32 %phi, %a
  store i32 %add2, i32 addrspace(1)* %out
  ret void
}

; The two S_ADD instructions should write to different registers, since
; different threads will take different control flow paths.

; SI-LABEL: {{^}}sgpr_if_else_valu_br:
; SI: s_add_i32 [[SGPR:s[0-9]+]]
; SI-NOT: s_add_i32 [[SGPR]]

define amdgpu_kernel void @sgpr_if_else_valu_br(i32 addrspace(1)* %out, float %a, i32 %b, i32 %c, i32 %d, i32 %e) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %tid_f = uitofp i32 %tid to float
  %tmp1 = fcmp ueq float %tid_f, 0.0
  br i1 %tmp1, label %if, label %else

if:
  %tmp2 = add i32 %b, %c
  br label %endif

else:
  %tmp3 = add i32 %d, %e
  br label %endif

endif:
  %tmp4 = phi i32 [%tmp2, %if], [%tmp3, %else]
  store i32 %tmp4, i32 addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}sgpr_if_else_valu_cmp_phi_br:

; SI: ; %else
; SI:      buffer_load_dword  [[AVAL:v[0-9]+]]
; SI:      v_cmp_gt_i32_e64   [[PHI:s\[[0-9]+:[0-9]+\]]], 0, [[AVAL]]

; SI: ; %if
; SI:      buffer_load_dword  [[AVAL:v[0-9]+]]
; SI:      v_cmp_eq_u32_e32   [[CMP_ELSE:vcc]], 0, [[AVAL]]
; SI-DAG:  s_andn2_b64        [[PHI]], [[PHI]], exec
; SI-DAG:  s_and_b64          [[TMP:s\[[0-9]+:[0-9]+\]]], [[CMP_ELSE]], exec
; SI:      s_or_b64           [[PHI]], [[PHI]], [[TMP]]

; SI: ; %endif
; SI:      v_cndmask_b32_e64  [[RESULT:v[0-9]+]], 0, -1, [[PHI]]
; SI:      buffer_store_dword [[RESULT]],
define amdgpu_kernel void @sgpr_if_else_valu_cmp_phi_br(i32 addrspace(1)* %out, i32 addrspace(1)* %a, i32 addrspace(1)* %b) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %tmp1 = icmp eq i32 %tid, 0
  br i1 %tmp1, label %if, label %else

if:
  %gep.if = getelementptr i32, i32 addrspace(1)* %a, i32 %tid
  %a.val = load i32, i32 addrspace(1)* %gep.if
  %cmp.if = icmp eq i32 %a.val, 0
  br label %endif

else:
  %gep.else = getelementptr i32, i32 addrspace(1)* %b, i32 %tid
  %b.val = load i32, i32 addrspace(1)* %gep.else
  %cmp.else = icmp slt i32 %b.val, 0
  br label %endif

endif:
  %tmp4 = phi i1 [%cmp.if, %if], [%cmp.else, %else]
  %ext = sext i1 %tmp4 to i32
  store i32 %ext, i32 addrspace(1)* %out
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0

attributes #0 = { readnone }

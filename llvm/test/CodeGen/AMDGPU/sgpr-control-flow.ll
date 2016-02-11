; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs< %s | FileCheck -check-prefix=SI %s
;
;
; Most SALU instructions ignore control flow, so we need to make sure
; they don't overwrite values from other blocks.

; If the branch decision is made based on a value in an SGPR then all
; threads will execute the same code paths, so we don't need to worry
; about instructions in different blocks overwriting each other.
; SI-LABEL: {{^}}sgpr_if_else_salu_br:
; SI: s_add
; SI: s_add

define void @sgpr_if_else_salu_br(i32 addrspace(1)* %out, i32 %a, i32 %b, i32 %c, i32 %d, i32 %e) {
entry:
  %0 = icmp eq i32 %a, 0
  br i1 %0, label %if, label %else

if:
  %1 = add i32 %b, %c
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

; The two S_ADD instructions should write to different registers, since
; different threads will take different control flow paths.

; SI-LABEL: {{^}}sgpr_if_else_valu_br:
; SI: s_add_i32 [[SGPR:s[0-9]+]]
; SI-NOT: s_add_i32 [[SGPR]]

define void @sgpr_if_else_valu_br(i32 addrspace(1)* %out, float %a, i32 %b, i32 %c, i32 %d, i32 %e) {
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

; FIXME: Should write to different SGPR pairs instead of copying to
; VALU for i1 phi.

; SI-LABEL: {{^}}sgpr_if_else_valu_cmp_phi_br:
; SI: buffer_load_dword [[AVAL:v[0-9]+]]
; SI: v_cmp_gt_i32_e32 [[CMP_IF:vcc]], 0, [[AVAL]]
; SI: v_cndmask_b32_e64 [[V_CMP:v[0-9]+]], 0, -1, [[CMP_IF]]

; SI: BB2_1:
; SI: buffer_load_dword [[AVAL:v[0-9]+]]
; SI: v_cmp_eq_i32_e32 [[CMP_ELSE:vcc]], 0, [[AVAL]]
; SI: v_cndmask_b32_e64 [[V_CMP]], 0, -1, [[CMP_ELSE]]

; SI: v_cmp_ne_i32_e32 [[CMP_CMP:vcc]], 0, [[V_CMP]]
; SI: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, -1, [[CMP_CMP]]
; SI: buffer_store_dword [[RESULT]]
define void @sgpr_if_else_valu_cmp_phi_br(i32 addrspace(1)* %out, i32 addrspace(1)* %a, i32 addrspace(1)* %b) {
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

; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=BOTH %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=VI -check-prefix=BOTH %s

; BOTH-LABEL: {{^}}s_rotl_i64:
; BOTH-DAG: s_lshl_b64
; BOTH-DAG: s_sub_i32
; BOTH-DAG: s_lshr_b64
; BOTH: s_or_b64
; BOTH: s_endpgm
define amdgpu_kernel void @s_rotl_i64(ptr addrspace(1) %in, i64 %x, i64 %y) {
entry:
  %0 = shl i64 %x, %y
  %1 = sub i64 64, %y
  %2 = lshr i64 %x, %1
  %3 = or i64 %0, %2
  store i64 %3, ptr addrspace(1) %in
  ret void
}

; BOTH-LABEL: {{^}}v_rotl_i64:
; SI-DAG: v_lshl_b64
; VI-DAG: v_lshlrev_b64
; BOTH-DAG: v_sub_{{[iu]}}32
; SI: v_lshr_b64
; VI: v_lshrrev_b64
; BOTH: v_or_b32
; BOTH: v_or_b32
; BOTH: s_endpgm
define amdgpu_kernel void @v_rotl_i64(ptr addrspace(1) %in, ptr addrspace(1) %xptr, ptr addrspace(1) %yptr) {
entry:
  %x = load i64, ptr addrspace(1) %xptr, align 8
  %y = load i64, ptr addrspace(1) %yptr, align 8
  %tmp0 = shl i64 %x, %y
  %tmp1 = sub i64 64, %y
  %tmp2 = lshr i64 %x, %tmp1
  %tmp3 = or i64 %tmp0, %tmp2
  store i64 %tmp3, ptr addrspace(1) %in, align 8
  ret void
}

; RUN: llc -mtriple=r600 -mcpu=cypress -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s


; FUNC-LABEL: {{^}}sext_in_reg_i1_i32_add:

; EG: MEM_{{.*}} STORE_{{.*}} [[RES:T[0-9]+\.[XYZW]]], [[ADDR:T[0-9]+.[XYZW]]]
; EG: SUB_INT {{[* ]*}}[[RES]]
; EG-NOT: BFE
define amdgpu_kernel void @sext_in_reg_i1_i32_add(ptr addrspace(1) %out, i1 %a, i32 %b) {
  %sext = sext i1 %a to i32
  %res = add i32 %b, %sext
  store i32 %res, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}sext_in_reg_i1_i32_sub:

; EG: MEM_{{.*}} STORE_{{.*}} [[RES:T[0-9]+\.[XYZW]]], [[ADDR:T[0-9]+.[XYZW]]]
; EG: ADD_INT {{[* ]*}}[[RES]]
; EG-NOT: BFE
define amdgpu_kernel void @sext_in_reg_i1_i32_sub(ptr addrspace(1) %out, i1 %a, i32 %b) {
  %sext = sext i1 %a to i32
  %res = sub i32 %b, %sext
  store i32 %res, ptr addrspace(1) %out
  ret void
}

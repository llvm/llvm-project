; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck  %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck  %s
; RUN: llc -march=r600 -mcpu=redwood -verify-machineinstrs < %s | FileCheck %s

; ALL-NOT: {{^}}func:
define internal i32 @func(i32 %a) {
entry:
  %tmp0 = add i32 %a, 1
  ret i32 %tmp0
}

; ALL: {{^}}kernel:
; GCN-NOT: s_swappc_b64
define amdgpu_kernel void @kernel(i32 addrspace(1)* %out) {
entry:
  %tmp0 = call i32 @func(i32 1)
  store i32 %tmp0, i32 addrspace(1)* %out
  ret void
}

; CHECK-NOT: func_alias
; ALL-NOT: func_alias
@func_alias = alias i32 (i32), i32 (i32)* @func

; ALL: {{^}}kernel3:
; GCN-NOT: s_swappc_b64
define amdgpu_kernel void @kernel3(i32 addrspace(1)* %out) {
entry:
  %tmp0 = call i32 @func_alias(i32 1)
  store i32 %tmp0, i32 addrspace(1)* %out
  ret void
}

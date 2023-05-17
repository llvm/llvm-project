; RUN: llc -mtriple amdgcn-unknown-linux-gnu -mcpu=tahiti -verify-machineinstrs < %s | FileCheck  %s
; RUN: llc -mtriple amdgcn-unknown-linux-gnu -mcpu=tonga -verify-machineinstrs < %s | FileCheck  %s
; RUN: llc -mtriple r600-unknown-linux-gnu -mcpu=redwood -verify-machineinstrs < %s | FileCheck %s --check-prefix=R600

; ALL-NOT: {{^}}func:
define internal i32 @func(i32 %a) {
entry:
  %tmp0 = add i32 %a, 1
  ret i32 %tmp0
}

; CHECK: {{^}}kernel:
; GCN-NOT: s_swappc_b64
define amdgpu_kernel void @kernel(ptr addrspace(1) %out) {
entry:
  %tmp0 = call i32 @func(i32 1)
  store i32 %tmp0, ptr addrspace(1) %out
  ret void
}

; CHECK: func_alias
; R600-NOT: func_alias
@func_alias = alias i32 (i32), ptr @func

; CHECK-NOT: {{^}}kernel3:
; GCN-NOT: s_swappc_b64
; R600: {{^}}kernel3:
define amdgpu_kernel void @kernel3(ptr addrspace(1) %out) {
entry:
  %tmp0 = call i32 @func_alias(i32 1)
  store i32 %tmp0, ptr addrspace(1) %out
  ret void
}

; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck  %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck  %s
; RUN: llc -march=r600 -mcpu=redwood -verify-machineinstrs < %s | FileCheck %s

; CHECK-NOT: {{^}}func:
define internal fastcc i32 @func(i32 %a) {
entry:
  %tmp0 = add i32 %a, 1
  ret i32 %tmp0
}

; CHECK: {{^}}kernel:
; CHECK-NOT: call
define amdgpu_kernel void @kernel(i32 addrspace(1)* %out) {
entry:
  %tmp0 = call i32 @func(i32 1)
  store i32 %tmp0, i32 addrspace(1)* %out
  ret void
}

; CHECK: {{^}}kernel2:
; CHECK-NOT: call
define amdgpu_kernel void @kernel2(i32 addrspace(1)* %out) {
entry:
  call void @kernel(i32 addrspace(1)* %out)
  ret void
}

; CHECK-NOT: func_alias
@func_alias = alias i32 (i32), i32 (i32)* @func

; CHECK: {{^}}kernel3:
; CHECK-NOT: call
define amdgpu_kernel void @kernel3(i32 addrspace(1)* %out) {
entry:
  %tmp0 = call i32 @func_alias(i32 1)
  store i32 %tmp0, i32 addrspace(1)* %out
  ret void
}

; CHECK-NOT: kernel_alias
@kernel_alias = alias void (i32 addrspace(1)*), void (i32 addrspace(1)*)* @kernel

; CHECK: {{^}}kernel4:
; CHECK-NOT: call
define amdgpu_kernel void @kernel4(i32 addrspace(1)* %out) {
entry:
  call void @kernel_alias(i32 addrspace(1)* %out)
  ret void
}

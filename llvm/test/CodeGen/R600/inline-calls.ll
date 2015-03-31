; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck  %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck  %s
; RUN: llc -march=r600 -mcpu=redwood -verify-machineinstrs < %s | FileCheck %s

; CHECK-NOT: {{^}}func:
define internal fastcc i32 @func(i32 %a) {
entry:
  %tmp0 = add i32 %a, 1
  ret i32 %tmp0
}

; CHECK: {{^}}kernel:
define void @kernel(i32 addrspace(1)* %out) {
entry:
  %tmp0 = call i32 @func(i32 1)
  store i32 %tmp0, i32 addrspace(1)* %out
  ret void
}

; CHECK: {{^}}kernel2:
define void @kernel2(i32 addrspace(1)* %out) {
entry:
  call void @kernel(i32 addrspace(1)* %out)
  ret void
}

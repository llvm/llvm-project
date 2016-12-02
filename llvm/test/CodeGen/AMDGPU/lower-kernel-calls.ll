; RUN: opt -amd-lower-kernel-calls -mtriple=amdgcn-- -mcpu=fiji -S < %s | FileCheck %s
define amdgpu_kernel void @test_kernel_to_call(i32 addrspace(1)* %p) #0 {
entry:
  store i32 2, i32 addrspace(1)* %p, align 4
  ret void
}

; Function Attrs: nounwind
define amdgpu_kernel void @test_call_kernel(i32 addrspace(1)* %p) #0 {
entry:
  store i32 1, i32 addrspace(1)* %p, align 4
; CHECK: call void @__amdgpu_test_kernel_to_call_kernel_body(i32 addrspace(1)* %p)
  call amdgpu_kernel void @test_kernel_to_call(i32 addrspace(1)* %p)
  ret void
}

; CHECK: define void @__amdgpu_test_kernel_to_call_kernel_body(i32 addrspace(1)* %p) #0 {
; CHECK: entry:
; CHECK:   store i32 2, i32 addrspace(1)* %p, align 4
; CHECK:   ret void
; CHECK: }

attributes #0 = { nounwind }

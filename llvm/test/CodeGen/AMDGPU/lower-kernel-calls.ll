; RUN: opt -amdgpu-lower-kernel-calls -mtriple=amdgcn--amdhsa -mcpu=fiji -S < %s | FileCheck %s
define amdgpu_kernel void @test_kernel_to_call(i32 addrspace(1)* %p) #0 {
entry:
  store i32 2, i32 addrspace(1)* %p, align 4
  ret void
}

declare amdgpu_kernel void @test_kernel_to_call_decl(i32 addrspace(1)* %p) #0

; Function Attrs: nounwind
define amdgpu_kernel void @test_call_kernel(i32 addrspace(1)* %p) #0 {
entry:
  store i32 1, i32 addrspace(1)* %p, align 4
  call amdgpu_kernel void @test_kernel_to_call(i32 addrspace(1)* %p)
; CHECK: call void @__amdgpu_test_kernel_to_call_kernel_body(i32 addrspace(1)* %p)
; CHECK-NOT: call amdgpu_kernel void @test_kernel_to_call(i32 addrspace(1)* %p)
  call amdgpu_kernel void @test_kernel_to_call_decl(i32 addrspace(1)* %p)
; CHECK: call void @__amdgpu_test_kernel_to_call_decl_kernel_body(i32 addrspace(1)* %p)
; CHECK-NOT: call amdgpu_kernel void @test_kernel_to_call_decl(i32 addrspace(1)* %p)
  ret void
}

; CHECK: define internal void @__amdgpu_test_kernel_to_call_kernel_body(i32 addrspace(1)* %p) #0
; CHECK:   store i32 2, i32 addrspace(1)* %p, align 4
; CHECK:   ret void

; CHECK: declare void @__amdgpu_test_kernel_to_call_decl_kernel_body(i32 addrspace(1)*)

attributes #0 = { nounwind }

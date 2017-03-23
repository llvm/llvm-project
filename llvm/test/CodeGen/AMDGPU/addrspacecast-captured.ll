; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -mcpu=kaveri -amdgpu-promote-alloca < %s | FileCheck %s

; Nothing should be done if the addrspacecast is captured.

declare void @consume_ptr2int(i32) #0

; CHECK-LABEL: @addrspacecast_captured(
; CHECK: %data = alloca i32, align 4
; CHECK: %cast = addrspacecast i32* %data to i32 addrspace(4)*
; CHECK: %ptr2int = ptrtoint i32 addrspace(4)* %cast to i32
; CHECK: store i32 %ptr2int, i32 addrspace(1)* %out
define amdgpu_kernel void @addrspacecast_captured(i32 addrspace(1)* %out) #0 {
entry:
  %data = alloca i32, align 4
  %cast = addrspacecast i32* %data to i32 addrspace(4)*
  %ptr2int = ptrtoint i32 addrspace(4)* %cast to i32
  store i32 %ptr2int, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: @addrspacecast_captured_store(
; CHECK: %data = alloca i32, align 4
; CHECK: %cast = addrspacecast i32* %data to i32 addrspace(4)*
; CHECK: store i32 addrspace(4)* %cast, i32 addrspace(4)* addrspace(1)* %out
define amdgpu_kernel void @addrspacecast_captured_store(i32 addrspace(4)* addrspace(1)* %out) #0 {
entry:
  %data = alloca i32, align 4
  %cast = addrspacecast i32* %data to i32 addrspace(4)*
  store i32 addrspace(4)* %cast, i32 addrspace(4)* addrspace(1)* %out
  ret void
}

; CHECK-LABEL: @addrspacecast_captured_call(
; CHECK: %data = alloca i32, align 4
; CHECK: %cast = addrspacecast i32* %data to i32 addrspace(4)*
; CHECK: %ptr2int = ptrtoint i32 addrspace(4)* %cast to i32
; CHECK: call void @consume_ptr2int(i32 %ptr2int)
define amdgpu_kernel void @addrspacecast_captured_call() #0 {
entry:
  %data = alloca i32, align 4
  %cast = addrspacecast i32* %data to i32 addrspace(4)*
  %ptr2int = ptrtoint i32 addrspace(4)* %cast to i32
  call void @consume_ptr2int(i32 %ptr2int)
  ret void
}

attributes #0 = { nounwind }

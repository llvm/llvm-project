; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -mcpu=kaveri -passes=amdgpu-promote-alloca < %s | FileCheck %s
; Nothing should be done if the addrspacecast is captured.

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"

declare void @consume_ptr2int(i32) #0

; CHECK-LABEL: @addrspacecast_captured(
; CHECK: %data = alloca i32, align 4, addrspace(5)
; CHECK: %cast = addrspacecast ptr addrspace(5) %data to ptr
; CHECK: %ptr2int = ptrtoint ptr %cast to i32
; CHECK: store i32 %ptr2int, ptr addrspace(1) %out
define amdgpu_kernel void @addrspacecast_captured(ptr addrspace(1) %out) #0 {
entry:
  %data = alloca i32, align 4, addrspace(5)
  %cast = addrspacecast ptr addrspace(5) %data to ptr
  %ptr2int = ptrtoint ptr %cast to i32
  store i32 %ptr2int, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @addrspacecast_captured_store(
; CHECK: %data = alloca i32, align 4, addrspace(5)
; CHECK: %cast = addrspacecast ptr addrspace(5) %data to ptr
; CHECK: store ptr %cast, ptr addrspace(1) %out
define amdgpu_kernel void @addrspacecast_captured_store(ptr addrspace(1) %out) #0 {
entry:
  %data = alloca i32, align 4, addrspace(5)
  %cast = addrspacecast ptr addrspace(5) %data to ptr
  store ptr %cast, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @addrspacecast_captured_call(
; CHECK: %data = alloca i32, align 4, addrspace(5)
; CHECK: %cast = addrspacecast ptr addrspace(5) %data to ptr
; CHECK: %ptr2int = ptrtoint ptr %cast to i32
; CHECK: call void @consume_ptr2int(i32 %ptr2int)
define amdgpu_kernel void @addrspacecast_captured_call() #0 {
entry:
  %data = alloca i32, align 4, addrspace(5)
  %cast = addrspacecast ptr addrspace(5) %data to ptr
  %ptr2int = ptrtoint ptr %cast to i32
  call void @consume_ptr2int(i32 %ptr2int)
  ret void
}

attributes #0 = { nounwind }

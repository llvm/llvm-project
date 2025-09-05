; RUN: opt < %s -S -nvptx-lower-alloca -infer-address-spaces | FileCheck %s
; RUN: opt < %s -S -nvptx-lower-alloca | FileCheck %s --check-prefix LOWERALLOCAONLY
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_35 | FileCheck %s --check-prefix PTX
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_35 | %ptxas-verify %}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-unknown-unknown"

define ptx_kernel void @kernel() {
; CHECK-LABEL: @kernel
; PTX-LABEL: .visible .entry kernel(
  %A = alloca i32
; CHECK: %A1 = alloca i32, align 4, addrspace(5)
; CHECK: store i32 0, ptr addrspace(5) %A1
; LOWERALLOCAONLY: %A1 = alloca i32, align 4, addrspace(5)
; LOWERALLOCAONLY: [[V2:%.*]] = addrspacecast ptr addrspace(5) %A1 to ptr
; LOWERALLOCAONLY: store i32 0, ptr [[V2]], align 4
; PTX: st.local.b32 [%SPL], 0
  store i32 0, ptr %A
  call void @callee(ptr %A)
  ret void
}

define void @alloca_in_explicit_local_as() {
; CHECK-LABEL: @alloca_in_explicit_local_as
; PTX-LABEL: .visible .func alloca_in_explicit_local_as(
  %A = alloca i32, addrspace(5)
; CHECK: store i32 0, ptr addrspace(5) %A, align 4
; PTX: st.local.b32 [%SPL], 0
; LOWERALLOCAONLY: %A = alloca i32, align 4, addrspace(5)
; LOWERALLOCAONLY: store i32 0, ptr addrspace(5) %A
  store i32 0, ptr addrspace(5) %A
  call void @callee(ptr addrspace(5) %A)
  ret void
}

declare void @callee(ptr)
declare void @callee_addrspace5(ptr addrspace(5))

!nvvm.annotations = !{!1}
!1 = !{ptr @alloca_in_explicit_local_as, !"alloca_in_explicit_local_as", i32 1}

; RUN: opt < %s -S -nvptx-lower-alloca -infer-address-spaces | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_35 | FileCheck %s --check-prefix PTX
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_35 | %ptxas-verify -arch=sm_35 %}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-unknown-unknown"

define void @kernel() {
; LABEL: @lower_alloca
; PTX-LABEL: .visible .entry kernel(
  %A = alloca i32
; CHECK: addrspacecast i32* %A to i32 addrspace(5)*
; CHECK: store i32 0, i32 addrspace(5)* {{%.+}}
; PTX: st.local.u32 [{{%rd[0-9]+}}], {{%r[0-9]+}}
  store i32 0, i32* %A
  call void @callee(i32* %A)
  ret void
}

declare void @callee(i32*)

!nvvm.annotations = !{!0}
!0 = !{void ()* @kernel, !"kernel", i32 1}

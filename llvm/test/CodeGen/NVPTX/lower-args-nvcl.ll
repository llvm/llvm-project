; RUN: opt < %s -S -nvptx-lower-args | FileCheck %s --check-prefixes COMMON,IR
; RUN: llc < %s -mcpu=sm_20 | FileCheck %s --check-prefixes COMMON,PTX
; RUN: %if ptxas %{ llc < %s -mcpu=sm_20 | %ptxas-verify %}

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-nvcl"

; COMMON-LABEL: ptr_nongeneric
define ptx_kernel void @ptr_nongeneric(ptr addrspace(1) %out, ptr addrspace(3) %in) {
; IR-NOT: addrspacecast
; PTX-NOT: cvta.to.global
; PTX:  ld.shared.b32
; PTX   st.global.b32
  %v = load i32, ptr addrspace(3) %in, align 4
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}

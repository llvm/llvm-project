; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_80 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_80 | %ptxas-verify %}

; Integer kernel parameters wider than 64 bits have no PTX fundamental type,
; so they must be passed as byte arrays, same as i128 and wider types.
; https://github.com/llvm/llvm-project/issues/226255

define ptx_kernel void @kernel_i65(i65 %p) {
; CHECK-LABEL: .visible .entry kernel_i65(
; CHECK: .param .align 16 .b8 kernel_i65_param_0[9]
  ret void
}

define ptx_kernel void @kernel_i96(i96 %p) {
; CHECK-LABEL: .visible .entry kernel_i96(
; CHECK: .param .align 16 .b8 kernel_i96_param_0[12]
  ret void
}

define ptx_kernel void @kernel_i127(i127 %p) {
; CHECK-LABEL: .visible .entry kernel_i127(
; CHECK: .param .align 16 .b8 kernel_i127_param_0[16]
  ret void
}

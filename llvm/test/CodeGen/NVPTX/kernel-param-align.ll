; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_60 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_60 | %ptxas -arch=sm_60 - %}

%struct.Large = type { [16 x double] }

; CHECK-LABEL: .entry func_align(
; CHECK: .param .u64 .ptr         .align 1  func_align_param_0
; CHECK: .param .u64 .ptr         .align 2  func_align_param_1
; CHECK: .param .u64 .ptr .global .align 4  func_align_param_2
; CHECK: .param .u64 .ptr .shared .align 8  func_align_param_3
; CHECK: .param .u64 .ptr .const  .align 16 func_align_param_4
; CHECK: .param .u64 .ptr .local  .align 32 func_align_param_5
define ptx_kernel void @func_align(ptr nocapture readonly align 1 %input,
                        ptr nocapture align 2 %out,
                        ptr addrspace(1) align 4 %global,
                        ptr addrspace(3) align 8 %shared,
                        ptr addrspace(4) align 16 %const,
                        ptr addrspace(5) align 32 %local) {
entry:
  ret void
}

; CHECK-LABEL: .entry func_noalign(
; CHECK: .param .u64 .ptr         .align 1 func_noalign_param_0
; CHECK: .param .u64 .ptr         .align 1 func_noalign_param_1
; CHECK: .param .u64 .ptr .global .align 1 func_noalign_param_2
; CHECK: .param .u64 .ptr .shared .align 1 func_noalign_param_3
; CHECK: .param .u64 .ptr .const  .align 1 func_noalign_param_4
; CHECK: .param .u64 .ptr .local  .align 1 func_noalign_param_5
define ptx_kernel void @func_noalign(ptr nocapture readonly %input,
                          ptr nocapture %out,
                          ptr addrspace(1) %global,
                          ptr addrspace(3) %shared,
                          ptr addrspace(4) %const,
                          ptr addrspace(5) %local) {
entry:
  ret void
}

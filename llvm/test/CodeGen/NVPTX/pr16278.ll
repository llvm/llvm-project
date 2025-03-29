; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

@one_f = addrspace(4) global float 1.000000e+00, align 4

define float @foo() {
; CHECK: ld.const.f32
  %val = load float, ptr addrspace(4) @one_f
  ret float %val
}

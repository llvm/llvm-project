; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_90 -verify-machineinstrs | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_90 -verify-machineinstrs | %ptxas-verify -arch=sm_90 %}

; PTX only has 8, 16, 32, and 64-bit integer types, so a value whose width is
; not one of those is named by the type of the storage it occupies.

; CHECK: .visible .global .align 1 .u8 g2;
@g2 = addrspace(1) global i2 0

; CHECK: .visible .global .align 4 .u32 g24;
@g24 = addrspace(1) global i24 0

; An i1 is stored as a byte. It must not be declared as .pred, which is only
; valid in the register state space.

; CHECK: .extern .global .align 1 .u8 g1;
@g1 = external addrspace(1) global i1

; Kernel parameters keep their storage size, which is part of the launch ABI.

; CHECK-LABEL: .visible .entry kernel(
; CHECK-NEXT:    .param .b8 kernel_param_0,
; CHECK-NEXT:    .param .b32 kernel_param_1,
; CHECK-NEXT:    .param .b64 kernel_param_2
define ptx_kernel void @kernel(i2 %a, i24 %b, i48 %c) {
  ret void
}

; Device function parameters are promoted to the PTX ABI's 32-bit minimum.

; CHECK-LABEL: .visible .func device(
; CHECK-NEXT:    .param .b32 device_param_0,
; CHECK-NEXT:    .param .b32 device_param_1,
; CHECK-NEXT:    .param .b64 device_param_2
define void @device(i2 %a, i24 %b, i48 %c) {
  ret void
}

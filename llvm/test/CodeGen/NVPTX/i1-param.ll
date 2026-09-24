; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx-nvidia-cuda"

; Make sure sub-byte integer operands to kernels get expanded out to .u8.

; CHECK: .entry foo
; CHECK:   .param .u8 foo_param_0,
; CHECK:   .param .u8 foo_param_1,
; CHECK:   .param .u8 foo_param_2,
; CHECK:   .param .u8 foo_param_3,
; CHECK:   .param .u8 foo_param_4,
; CHECK:   .param .u8 foo_param_5,
; CHECK:   .param .u8 foo_param_6,
; CHECK:   .param .u64 .ptr .align 1 foo_param_7
define ptx_kernel void @foo(i1 %p, i2 %p2, i3 %p3, i4 %p4, i5 %p5, i6 %p6,
                            i7 %p7, ptr %out) {
  %val = zext i1 %p to i32
  store i32 %val, ptr %out
  ret void
}

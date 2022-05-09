; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx -mcpu=sm_20 | %ptxas-verify %}

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx-nvidia-cuda"

; Make sure predicate (i1) operands to kernels get expanded out to .u8

; CHECK: .entry foo
; CHECK:   .param .u8 foo_param_0
; CHECK:   .param .u32 foo_param_1
define void @foo(i1 %p, i32* %out) {
  %val = zext i1 %p to i32
  store i32 %val, i32* %out
  ret void
}


!nvvm.annotations = !{!0}
!0 = !{void (i1, i32*)* @foo, !"kernel", i32 1}

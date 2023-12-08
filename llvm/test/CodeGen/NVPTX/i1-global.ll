; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx-nvidia-cuda"

; CHECK: .visible .global .align 1 .u8 mypred
@mypred = addrspace(1) global i1 true, align 1


define void @foo(i1 %p, ptr %out) {
  %ld = load i1, ptr addrspace(1) @mypred
  %val = zext i1 %ld to i32
  store i32 %val, ptr %out
  ret void
}


!nvvm.annotations = !{!0}
!0 = !{ptr @foo, !"kernel", i32 1}

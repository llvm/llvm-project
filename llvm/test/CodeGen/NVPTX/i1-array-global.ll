; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx-nvidia-cuda"

@global_cst = private constant [6 x i1] [i1 true, i1 false, i1 true, i1 false, i1 true, i1 false]

; CHECK: .global .align 1 .b8 global_cst[6] = {1, 0, 1, 0, 1}
define void @kernel(i32 %i, ptr %out) {
  %5 = getelementptr inbounds i1, ptr @global_cst, i32 %i
  %6 = load i1, ptr %5, align 1
  store i1 %6, ptr %out, align 1
  ret void
}

!nvvm.annotations = !{!0}
!0 = !{ptr @kernel, !"kernel", i32 1}


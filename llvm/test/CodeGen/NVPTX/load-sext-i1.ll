; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx-nvidia-cuda"

define void @main(ptr %a1, i32 %a2, ptr %arg3) {
; CHECK: ld.u8
; CHECK-NOT: ld.u1
  %t1 = getelementptr i1, ptr %a1, i32 %a2
  %t2 = load i1, ptr %t1
  %t3 = sext i1 %t2 to i32
  store i32 %t3, ptr %arg3
  ret void
}

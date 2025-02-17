; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; CHECK-LABEL: _Z3foobbbPb
define void @_Z3foobbbPb(i1 zeroext %p1, i1 zeroext %p2, i1 zeroext %p3, ptr nocapture %output) {
entry:
; CHECK: selp.b32       %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %p{{[0-9]+}}
  %.sink.v = select i1 %p1, i1 %p2, i1 %p3
  %frombool5 = zext i1 %.sink.v to i8
  store i8 %frombool5, ptr %output, align 1
  ret void
}

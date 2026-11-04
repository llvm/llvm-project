; RUN: llc %s -filetype=asm -o - | FileCheck %s

; CHECK: vmov.i8 d3, #0xff

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv8a-unknown-linux-gnueabihf"

; Function Attrs: mustprogress noimplicitfloat nounwind
define void @cvt_vec() local_unnamed_addr {
entry:
  tail call void asm sideeffect "", "{d3}"(<8 x i8> splat (i8 -1))
  ret void
}


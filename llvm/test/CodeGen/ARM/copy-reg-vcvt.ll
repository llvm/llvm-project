; RUN: llc -filetype=asm -O3  %s -o - | FileCheck %s
; CHECK:      vldr s0, [r0]
; CHECK-NEXT: vcvt.f64.f32 d1, s0

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv8a-unknown-linux-gnueabihf"

@a = local_unnamed_addr global float 0.000000e+00, align 4

; Function Attrs: mustprogress noimplicitfloat nounwind
define void @_Z1bv() local_unnamed_addr {
entry:
  %0 = load float, ptr @a, align 4
  tail call void asm sideeffect "", "{d1}"(float %0)
  ret void
}


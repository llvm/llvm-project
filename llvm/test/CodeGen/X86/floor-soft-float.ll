; RUN: llc < %s -mattr=+sse4.1,-avx | FileCheck %s --check-prefix=CHECK-HARD-FLOAT
; RUN: llc < %s -mattr=+sse4.1,-avx,+soft-float | FileCheck %s --check-prefix=CHECK-SOFT-FLOAT

target triple = "x86_64-unknown-linux-gnu"

declare float @llvm.floor.f32(float)

; CHECK-SOFT-FLOAT: callq floorf
; CHECK-HARD-FLOAT: roundss $9, %xmm0, %xmm0
define float @myfloor(float %a) {
  %val = tail call float @llvm.floor.f32(float %a)
  ret float %val
}

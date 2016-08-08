; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s

; Test case adapted from PR24216.

define void @foo(<16 x i8>* nocapture readonly %in, <16 x i8>* nocapture %out) {
entry:
  %0 = load <16 x i8>, <16 x i8>* %in, align 16
  %1 = shufflevector <16 x i8> %0, <16 x i8> undef, <16 x i32> <i32 2, i32 3, i32 4, i32 5, i32 2, i32 3, i32 4, i32 5, i32 2, i32 3, i32 4, i32 5, i32 2, i32 3, i32 4, i32 5>
  store <16 x i8> %1, <16 x i8>* %out, align 16
  ret void
}

; CHECK: vperm
; CHECK-NOT: vspltw

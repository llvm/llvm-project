; RUN: llc -mtriple=aarch64 -o - -O0 %s | FileCheck %s
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios5.0.0"

; This test checks we don't fail isel due to unhandled build_pair nodes.
; CHECK: bfi
define void @compare_and_swap128() {
  %1 = call i128 asm sideeffect "nop", "=r,~{memory}"()
  store i128 %1, i128* undef, align 16
  ret void
}



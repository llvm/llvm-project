; RUN: llvm-link -S %s %p/Inputs/type-unique-opaque.ll | FileCheck %s

; Test that a failed attempt at merging %u2 and %t2 (for the other file) will
; not cause %u and %t to get merged.

; CHECK: %u = type opaque
; CHECK: external global %u

%u = type opaque
%u2 = type { %u, i8 }

@g = external global %u

define ptr @test() {
  ret ptr @g
}

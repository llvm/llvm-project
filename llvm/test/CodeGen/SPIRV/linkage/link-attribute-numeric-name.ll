; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; A numeric-named function has no meaningful linkage name, so no
; LinkageAttributes decoration should be emitted for it.
; CHECK-NOT: OpDecorate {{.*}} LinkageAttributes

define void @0() {
  ret void
}

define spir_kernel void @kernel() {
  ret void
}

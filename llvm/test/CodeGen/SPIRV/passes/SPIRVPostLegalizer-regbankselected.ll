; RUN: llc -O0 -mtriple=spirv64-unknown-unknown -verify-machineinstrs --stop-after=spirv-postlegalizer %s -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown -verify-machineinstrs -enable-new-pm --stop-after=spirv-postlegalizer %s -o - | FileCheck %s

; The unused exponent still needs a register class once RegBankSelected is set.

; CHECK: regBankSelected: true
; CHECK: , %[[#]]:iid(i32) = G_FFREXP

define spir_func float @unused_exponent(float %x) {
  %r = call { float, i32 } @llvm.frexp.f32.i32(float %x)
  %m = extractvalue { float, i32 } %r, 0
  ret float %m
}

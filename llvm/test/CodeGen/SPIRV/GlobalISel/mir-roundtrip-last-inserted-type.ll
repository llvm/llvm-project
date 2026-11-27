; RUN: llc -O0 -mtriple=spirv64-unknown-unknown -verify-machineinstrs -stop-after=spirv-postlegalizer %s -o %t.mir
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown -verify-machineinstrs %s -o %t.ref.s
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown -verify-machineinstrs -start-after=spirv-postlegalizer %t.mir -o %t.s
; RUN: diff %t.ref.s %t.s

; Instruction selection creates the 2.0 constant after MIR load. It must go
; after the last inserted type, as in a direct compile.

define spir_func float @foo(float %a) {
  %r = fadd float %a, 2.0
  ret float %r
}

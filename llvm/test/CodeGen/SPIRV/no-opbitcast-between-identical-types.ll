; The goal of the test case is to ensure that no OpBitcast is generated for a bitcast between identical types.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpFunction
; CHECK-NO: OpBitcast
; CHECK: OpReturn

define void @foo() {
entry:
  %r = bitcast i32 0 to i32
  ret void
}

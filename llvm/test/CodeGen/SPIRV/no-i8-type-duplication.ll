; The goal of the test is to check that only one "OpTypeInt 8" instruction
; is generated for a series of LLVM integer types with number of bits less
; than 8.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV: %[[#CharTy:]] = OpTypeInt 8 0
; CHECK-SPIRV-NO: %[[#CharTy:]] = OpTypeInt 8 0

define spir_func void @foo(i2 %a, i4 %b) {
entry:
  ret void
}

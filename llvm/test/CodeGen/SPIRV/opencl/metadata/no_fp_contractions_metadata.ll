; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-DEFAULT
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-fp-contract=on %s -o - | FileCheck %s --check-prefix=CHECK-DEFAULT
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-fp-contract=off %s -o - | FileCheck %s --check-prefix=CHECK-DEFAULT
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-fp-contract=fast %s -o - | FileCheck %s --check-prefix=CHECK-FAST
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DEFAULT: OpEntryPoint Kernel %[[#ENTRY:]] "foo"
; CHECK-DEFAULT: OpExecutionMode %[[#ENTRY]] ContractionOff

; CHECK-FAST: OpEntryPoint Kernel %[[#ENTRY:]] "foo"
; CHECK-FAST-NOT: OpExecutionMode %[[#ENTRY]] ContractionOff
define spir_kernel void @foo() {
entry:
  ret void
}

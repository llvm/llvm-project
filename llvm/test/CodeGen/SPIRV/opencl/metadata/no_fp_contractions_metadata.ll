; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-ON
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-fp-contract=on %s -o - | FileCheck %s --check-prefix=CHECK-ON
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-fp-contract=off %s -o - | FileCheck %s --check-prefix=CHECK-OFF
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-fp-contract=fast %s -o - | FileCheck %s --check-prefix=CHECK-ON
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-fp-contract=off %s -o - -filetype=obj | spirv-val %}

; Nothing in @foo disables contraction; only =off forces ContractionOff.
; CHECK-ON: OpEntryPoint Kernel %[[#ENTRY:]] "foo"
; CHECK-ON-NOT: OpExecutionMode %[[#ENTRY]] ContractionOff

; CHECK-OFF: OpEntryPoint Kernel %[[#ENTRY:]] "foo"
; CHECK-OFF: OpExecutionMode %[[#ENTRY]] ContractionOff
define spir_kernel void @foo() {
entry:
  ret void
}

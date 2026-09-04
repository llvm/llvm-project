;; Check translation of (intel_)reqd_sub_group_size metadata to SubgroupSize
;; execution mode.

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV: OpCapability SubgroupDispatch
; CHECK-SPIRV: OpEntryPoint Kernel %[[#bar_kernel:]] "bar"
; CHECK-SPIRV: OpEntryPoint Kernel %[[#foo_kernel:]] "foo"
; CHECK-SPIRV: OpExecutionMode %[[#bar_kernel]] SubgroupSize 8
; CHECK-SPIRV: OpExecutionMode %[[#foo_kernel]] SubgroupSize 8

define spir_kernel void @bar() !reqd_sub_group_size !0 {
entry:
  ret void
}

define spir_kernel void @foo() !intel_reqd_sub_group_size !0 {
entry:
  ret void
}

!0 = !{i32 8}

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv64-vulkan1.6-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-vulkan1.6-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV: OpName %[[#FOO:]] "foo"
; CHECK-SPIRV: OpDecorate %[[#FOO]] LinkageAttributes "foo" Import

define spir_func void @use() {
entry:
  call spir_func void @foo()
  ret void
}

declare hidden spir_func void @foo()

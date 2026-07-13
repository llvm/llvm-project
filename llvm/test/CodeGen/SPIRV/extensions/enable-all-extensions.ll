; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=all %s -o - | FileCheck %s
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-amd-amdhsa %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=all %s -o - -filetype=obj | spirv-val %}

define i6 @getConstantI6() {
  ret i6 2
}

; CHECK: OpExtension "SPV_ALTERA_arbitrary_precision_integers"

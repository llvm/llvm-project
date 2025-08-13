; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - | spirv-as - -o - | spirv-val %}

%literal_32 = type target("spirv.Literal", 32)
%literal_true = type target("spirv.Literal", 1)

; CHECK-DAG: OpUnknown(21, 4) [[int_t:%[0-9]+]] 32 1
%int_t = type target("spirv.Type", %literal_32, %literal_true, 21, 4, 32)

; CHECK-DAG: {{%[0-9]+}} = OpTypeFunction [[int_t]]
define %int_t @foo() {
entry:
  %v = alloca %int_t
  %i = load %int_t, ptr %v

; CHECK-DAG: [[i:%[0-9]+]] = OpUndef [[int_t]]
; CHECK-DAG: OpReturnValue [[i]]
  ret %int_t %i
}

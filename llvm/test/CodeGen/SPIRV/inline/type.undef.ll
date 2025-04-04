; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - | spirv-as - -o - | spirv-val %}

%literal_32 = type target("spirv.Literal", 32)
%literal_true = type target("spirv.Literal", 1)

; CHECK-DAG: OpUnknown(21, 4) [[int_t:%[0-9]+]] 32 1
%int_t = type target("spirv.Type", %literal_32, %literal_true, 21, 4, 32)

; CHECK-DAG: {{%[0-9]+}} = OpTypeFunction [[int_t]]
define %int_t @foo() {
entry:
; CHECK-DAG: [[undef:%[0-9]+]] = OpUndef [[int_t]]
; CHECK-DAG: OpReturnValue [[undef]]
  ret %int_t undef
}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - | spirv-as - -o - | spirv-val %}

; CHECK-DAG: [[float_t:%[0-9]+]] = OpTypeFloat 32
; CHECK-DAG: [[uint32_t:%[0-9]+]] = OpTypeInt 32 0

; CHECK-DAG: [[image_t:%[0-9]+]] = OpTypeImage [[float_t]] 2D 2 0 0 1 Unknown
%type_2d_image = type target("spirv.Image", float, 1, 2, 0, 0, 1, 0)

%literal_true = type target("spirv.Literal", 1)
%literal_32 = type target("spirv.Literal", 32)

; CHECK-DAG: [[uint32_4:%[0-9]+]] = OpConstant [[uint32_t]] 4
%integral_constant_4 = type target("spirv.IntegralConstant", i32, 4)

; CHECK-DAG: OpUnknown(28, 4) [[array_t:%[0-9]+]] [[image_t]] [[uint32_4]]
%ArrayTex2D = type target("spirv.Type", %type_2d_image, %integral_constant_4, 28, 0, 0)

; CHECK-DAG: OpUnknown(21, 4) [[int_t:%[0-9]+]] 32 1
%int_t = type target("spirv.Type", %literal_32, %literal_true, 21, 0, 0)

; CHECK-DAG: [[getTexArray_t:%[0-9]+]] = OpTypeFunction [[array_t]]
; CHECK-DAG: [[getInt_t:%[0-9]+]] = OpTypeFunction [[int_t]]

; CHECK: [[getTexArray:%[0-9]+]] = OpFunction [[array_t]] None [[getTexArray_t]]
declare %ArrayTex2D @getTexArray()

; CHECK: [[getInt:%[0-9]+]] = OpFunction [[int_t]] None [[getInt_t]]
declare %int_t @getInt()

define void @main() #1 {
entry:
; CHECK: {{%[0-9]+}} = OpFunctionCall [[array_t]] [[getTexArray]]
  %retTex = call %ArrayTex2D @getTexArray()

; CHECK: {{%[0-9]+}} = OpFunctionCall [[int_t]] [[getInt]]
  %i = call %int_t @getInt()

  ret void
}

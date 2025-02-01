; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - | spirv-as - -o - | spirv-val %}

; CHECK: [[uint32_t:%[0-9]+]] = OpTypeInt 32 0

; CHECK: [[image_t:%[0-9]+]] = OpTypeImage %3 2D 2 0 0 1 Unknown
%type_2d_image = type target("spirv.Image", float, 1, 2, 0, 0, 1, 0)

%literal_false = type target("spirv.Literal", 0)
%literal_8 = type target("spirv.Literal", 8)

; CHECK: [[uint32_4:%[0-9]+]] = OpConstant [[uint32_t]] 4
%integral_constant_4 = type target("spirv.IntegralConstant", i32, 4)

; CHECK: !0x4001c [[array_t:%[0-9]+]] [[image_t]] [[uint32_4]]
%ArrayTex2D = type target("spirv.Type", %type_2d_image, %integral_constant_4, 28, 0, 0)

; CHECK: [[getTexArray_t:%[0-9]+]] = OpTypeFunction [[array_t]]

; CHECK: [[getTexArray:%[0-9]+]] = OpFunction [[array_t]] None [[getTexArray_t]]
declare %ArrayTex2D @getTexArray()

define void @main() #1 {
entry:
  %images = alloca %ArrayTex2D

; CHECK: {{%[0-9]+}} = OpFunctionCall [[array_t]] [[getTexArray]]
  %retTex = call %ArrayTex2D @getTexArray()

  ret void
}

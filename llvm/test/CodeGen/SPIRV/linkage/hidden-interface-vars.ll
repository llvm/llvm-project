; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan1.3-pixel %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-pixel %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; Check that interface variables (Input, Output, PushConstant) do not get
; Import linkage even when declared as external hidden.

; CHECK-NOT: OpCapability Linkage
; CHECK-NOT: LinkageAttributes

@input_var = external hidden addrspace(7) global <4 x float>
@output_var = external hidden addrspace(8) global <4 x float>

; CHECK-DAG: %[[#INPUT_TYPE:]] = OpTypePointer Input %[[#FLOAT4:]]
; CHECK-DAG: %[[#OUTPUT_TYPE:]] = OpTypePointer Output %[[#FLOAT4]]
; CHECK-DAG: %[[#INPUT_VAR:]] = OpVariable %[[#INPUT_TYPE]] Input
; CHECK-DAG: %[[#OUTPUT_VAR:]] = OpVariable %[[#OUTPUT_TYPE]] Output

define void @main() #0 {
entry:
  %val = load <4 x float>, ptr addrspace(7) @input_var
  store <4 x float> %val, ptr addrspace(8) @output_var
  ret void
}

attributes #0 = { "hlsl.shader"="pixel" }

; RUN: llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - | FileCheck %s

; The function in the test has external linkage because we want it to be 
; output. Since this test uses Linkage we cannot validate for Vulkan. That
; should not be a problem as it is only testing general shader behavior.
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#i32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#struct:]] = OpTypeStruct %[[#]] %[[#]] %[[#i32]]
; CHECK-DAG: %[[#ptr_struct:]] = OpTypePointer Function %[[#struct]]
; CHECK-DAG: %[[#ptr_i32:]] = OpTypePointer Function %[[#i32]]
; CHECK-DAG: %[[#idx:]] = OpConstant %[[#]] 2

%struct.VSOutput = type { <4 x float>, <2 x float>, i32 }

; CHECK: %[[#func:]] = OpFunction %[[#i32]] None %[[#]]
; CHECK: %[[#input:]] = OpFunctionParameter %[[#ptr_struct]]
define spir_func i32 @_Z13sampleTexture8VSOutput(ptr byval(%struct.VSOutput) align 1 %input) {
entry:
; CHECK: %[[#gep:]] = OpInBoundsAccessChain %[[#ptr_i32]] %[[#input]] %[[#idx]]
  %ColorOffset = getelementptr inbounds nuw i8, ptr %input, i64 24
  
; CHECK: %[[#val:]] = OpLoad %[[#i32]] %[[#gep]]
  %val = load i32, ptr %ColorOffset, align 1
  
; CHECK: OpReturnValue %[[#val]]
  ret i32 %val
}

define void @main() #0 {
entry:
  ret void
}

attributes #0 = { "hlsl.shader"="pixel" }

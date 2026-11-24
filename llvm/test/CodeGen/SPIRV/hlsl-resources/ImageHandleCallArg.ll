; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; An image handle passed to a call must be defined.

; CHECK-DAG: %[[#Float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#Image:]] = OpTypeImage %[[#Float]] Buffer 2 0 0 2 Unknown
; CHECK-DAG: %[[#ImagePtr:]] = OpTypePointer UniformConstant %[[#Image]]
; CHECK-DAG: %[[#Var:]] = OpVariable %[[#ImagePtr]] UniformConstant
; CHECK: %[[#Callee:]] = OpFunction
; CHECK: %[[#Handle:]] = OpLoad %[[#Image]] %[[#Var]]
; CHECK: OpFunctionCall %[[#]] %[[#Callee]] %[[#Handle]]

@.str = private unnamed_addr constant [4 x i8] c"Buf\00", align 1

define internal void @callee(target("spirv.Image", float, 5, 2, 0, 0, 2, 0) %h) {
entry:
  ret void
}

define void @main() #0 {
entry:
  %h = call target("spirv.Image", float, 5, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_5_2_0_0_2_0t(i32 0, i32 0, i32 1, i32 0, ptr @.str)
  call void @callee(target("spirv.Image", float, 5, 2, 0, 0, 2, 0) %h)
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

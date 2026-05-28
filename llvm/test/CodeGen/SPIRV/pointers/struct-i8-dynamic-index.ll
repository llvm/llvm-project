; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv1.6-unknown-vulkan1.3 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3 %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

%struct.S = type { [4 x <2 x float>] }

@G = internal global %struct.S zeroinitializer
@IDX = internal global i32 1
@OUT = internal global <2 x float> zeroinitializer

define void @main() #0 {
entry:
  %idx = load i32, ptr @IDX
  ; CHECK-DAG: %[[#IDX:]] = OpLoad %[[#]] %[[#]]
  ; CHECK-DAG: %[[#G:]] = OpVariable %[[#]] Function
  ; CHECK: %[[#AC:]] = OpAccessChain %[[#]] %[[#G]] %[[#]] %[[#IDX]]
  ; CHECK: %[[#VAL:]] = OpLoad %[[#]] %[[#AC]]
  ; CHECK: OpStore %[[#]] %[[#VAL]]
  %gep = getelementptr [8 x i8], ptr @G, i32 %idx
  %val = load <2 x float>, ptr %gep
  store <2 x float> %val, ptr @OUT
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

; RUN: llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute --spirv-ext=+SPV_EXT_long_vector %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute --spirv-ext=+SPV_EXT_long_vector %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; Vector16 is a Kernel-only capability. On Vulkan, 8 and 16 element vectors
; should be of type OpTypeVectorIdEXT rather than OpTypeVector.
; Short vectors should keep using OpTypeVector.

; CHECK-NOT: OpCapability Vector16
; CHECK-DAG: OpCapability LongVectorEXT
; CHECK-NOT: OpCapability Vector16
; CHECK-DAG: OpExtension "SPV_EXT_long_vector"

; CHECK-DAG: %[[#FLOAT:]] = OpTypeFloat 32
; CHECK-DAG: %[[#INT:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#C8:]] = OpConstant %[[#INT]] 8
; CHECK-DAG: %[[#C16:]] = OpConstant %[[#INT]] 16
; CHECK-DAG: %[[#V8:]] = OpTypeVectorIdEXT %[[#FLOAT]] %[[#C8]]
; CHECK-DAG: %[[#V16:]] = OpTypeVectorIdEXT %[[#FLOAT]] %[[#C16]]
; CHECK-DAG: %[[#V4:]] = OpTypeVector %[[#FLOAT]] 4
; CHECK-NOT: OpTypeVector %[[#FLOAT]] 8
; CHECK-NOT: OpTypeVector %[[#FLOAT]] 16

@g4 = internal addrspace(10) global <4 x float> zeroinitializer
@g8 = internal addrspace(10) global <8 x float> zeroinitializer
@g16 = internal addrspace(10) global <16 x float> zeroinitializer

define void @main() #0 {
entry:
  ; CHECK: OpFunction
  ; CHECK: %[[#L8:]] = OpLoad %[[#V8]]
  %l8 = load <8 x float>, ptr addrspace(10) @g8
  ; CHECK: %[[#L16:]] = OpLoad %[[#V16]]
  %l16 = load <16 x float>, ptr addrspace(10) @g16
  ; CHECK: %[[#A8:]] = OpFAdd %[[#V8]] %[[#L8]] %[[#L8]]
  %a8 = fadd <8 x float> %l8, %l8
  ; CHECK: %[[#A16:]] = OpFAdd %[[#V16]] %[[#L16]] %[[#L16]]
  %a16 = fadd <16 x float> %l16, %l16
  ; CHECK: %[[#S4:]] = OpVectorShuffle %[[#V4]] %[[#A8]] %[[#A8]] 0 1 2 3
  %s4 = shufflevector <8 x float> %a8, <8 x float> %a8, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ; CHECK: %[[#S8:]] = OpVectorShuffle %[[#V8]] %[[#A16]] %[[#A16]] 0 1 2 3 4 5 6 7
  %s8 = shufflevector <16 x float> %a16, <16 x float> %a16, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ; CHECK: OpStore %[[#]] %[[#S8]]
  store <8 x float> %s8, ptr addrspace(10) @g8
  ; CHECK: OpStore %[[#]] %[[#S4]]
  store <4 x float> %s4, ptr addrspace(10) @g4
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

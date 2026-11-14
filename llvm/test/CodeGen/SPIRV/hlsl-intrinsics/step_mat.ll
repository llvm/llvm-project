; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-NOT: OpCapability Vector16
; CHECK-DAG: OpCapability Float16
; CHECK-DAG: %[[#void:]] = OpTypeVoid
; CHECK-DAG: %[[#bool:]] = OpTypeBool
; CHECK-DAG: %[[#vec4bool:]] = OpTypeVector %[[#bool]] 4
; CHECK-DAG: %[[#vec2bool:]] = OpTypeVector %[[#bool]] 2
; CHECK-DAG: %[[#f32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#vec4f32:]] = OpTypeVector %[[#f32]] 4
; CHECK-DAG: %[[#vec2f32:]] = OpTypeVector %[[#f32]] 2
; CHECK-DAG: %[[#f16:]] = OpTypeFloat 16
; CHECK-DAG: %[[#vec4f16:]] = OpTypeVector %[[#f16]] 4

@shuffle_f32_4 = internal addrspace(10) global <4 x float> zeroinitializer
@wide_f32_6 = internal addrspace(10) global [6 x float] zeroinitializer
@wide_f16_9 = internal addrspace(10) global [9 x half] zeroinitializer
@wide_f32_16 = internal addrspace(10) global [16 x float] zeroinitializer

define internal void @step_float6_from_shuffle() {
entry:
  ; CHECK: OpFunction %[[#void]] None
  ; CHECK: %[[#shuffle_f32:]] = OpLoad %[[#vec4f32]]
  ; CHECK: OpCompositeExtract %[[#f32]] %[[#shuffle_f32]] 0
  ; CHECK: OpFOrdLessThan %[[#vec4bool]]
  ; CHECK: OpFOrdLessThan %[[#vec2bool]]
  ; CHECK: OpSelect %[[#vec4f32]]
  ; CHECK: OpSelect %[[#vec2f32]]
  %vec = load <4 x float>, ptr addrspace(10) @shuffle_f32_4
  %va = shufflevector <4 x float> %vec, <4 x float> %vec,
            <6 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5>
  %c = fcmp olt <6 x float> %va, %va
  %r = select <6 x i1> %c, <6 x float> zeroinitializer, <6 x float> splat (float 1.000000e+00)
  store <6 x float> %r, ptr addrspace(10) @wide_f32_6
  ret void
}

define internal void @step_half9() {
entry:
  ; CHECK: OpFunction %[[#void]] None
  ; CHECK-COUNT-2: OpFOrdLessThan %[[#vec4bool]]
  ; CHECK: OpFOrdLessThan %[[#bool]]
  ; CHECK-COUNT-2: OpSelect %[[#vec4f16]]
  ; CHECK: OpSelect %[[#f16]]
  %va = load <9 x half>, ptr addrspace(10) @wide_f16_9
  %vb = load <9 x half>, ptr addrspace(10) @wide_f16_9
  %c = fcmp olt <9 x half> %va, %vb
  %r = select <9 x i1> %c, <9 x half> zeroinitializer, <9 x half> splat (half 1.000000e+00)
  store <9 x half> %r, ptr addrspace(10) @wide_f16_9
  ret void
}

define internal void @step_float16() {
entry:
  ; CHECK: OpFunction %[[#void]] None
  ; CHECK-COUNT-4: OpFOrdLessThan %[[#vec4bool]]
  ; CHECK-COUNT-4: OpSelect %[[#vec4f32]]
  %va = load <16 x float>, ptr addrspace(10) @wide_f32_16
  %vb = load <16 x float>, ptr addrspace(10) @wide_f32_16
  %c = fcmp olt <16 x float> %va, %vb
  %r = select <16 x i1> %c, <16 x float> zeroinitializer, <16 x float> splat (float 1.000000e+00)
  store <16 x float> %r, ptr addrspace(10) @wide_f32_16
  ret void
}

define void @main() #0 {
entry:
  call void @step_float6_from_shuffle()
  call void @step_half9()
  call void @step_float16()
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

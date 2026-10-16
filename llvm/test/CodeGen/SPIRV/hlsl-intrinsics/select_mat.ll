; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-NOT: OpCapability Vector16
; CHECK-DAG: OpCapability Float16
; CHECK-DAG: OpCapability Int16
; CHECK-DAG: %[[#bool:]] = OpTypeBool
; CHECK-DAG: %[[#int_32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#int_16:]] = OpTypeInt 16 0
; CHECK-DAG: %[[#float_32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#float_16:]] = OpTypeFloat 16
; CHECK-DAG: %[[#vec4_bool:]] = OpTypeVector %[[#bool]] 4
; CHECK-DAG: %[[#vec4_int_32:]] = OpTypeVector %[[#int_32]] 4
; CHECK-DAG: %[[#vec4_int_16:]] = OpTypeVector %[[#int_16]] 4
; CHECK-DAG: %[[#vec4_float_32:]] = OpTypeVector %[[#float_32]] 4
; CHECK-DAG: %[[#vec4_float_16:]] = OpTypeVector %[[#float_16]] 4
; CHECK-DAG: %[[#vec2_bool:]] = OpTypeVector %[[#bool]] 2
; CHECK-DAG: %[[#vec2_int_32:]] = OpTypeVector %[[#int_32]] 2
; CHECK-DAG: %[[#vec2_float_32:]] = OpTypeVector %[[#float_32]] 2

@shuffle_f32_4 = internal addrspace(10) global <4 x float> zeroinitializer
@wide_f32_6 = internal addrspace(10) global [6 x float] zeroinitializer
@wide_f16_9 = internal addrspace(10) global [9 x half] zeroinitializer
@wide_f32_16 = internal addrspace(10) global [16 x float] zeroinitializer
@shuffle_i32_4 = internal addrspace(10) global <4 x i32> zeroinitializer
@wide_i32_6 = internal addrspace(10) global [6 x i32] zeroinitializer
@wide_i16_9 = internal addrspace(10) global [9 x i16] zeroinitializer
@wide_i32_9 = internal addrspace(10) global [9 x i32] zeroinitializer
@wide_i32_16 = internal addrspace(10) global [16 x i32] zeroinitializer

define internal void @select_float6_from_shuffle() {
entry:
  ; CHECK: OpFunction
  ; CHECK: %[[#shuffle_f32:]] = OpLoad %[[#vec4_float_32]]
  ; CHECK: OpCompositeExtract %[[#float_32]] %[[#shuffle_f32]] 0
  ; CHECK: OpSelect %[[#vec4_float_32]]
  ; CHECK: OpSelect %[[#vec2_float_32]]
  ; CHECK: OpFunctionEnd
  %vec = load <4 x float>, ptr addrspace(10) @shuffle_f32_4
  %va = shufflevector <4 x float> %vec, <4 x float> %vec,
            <6 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5>
  %vc = load <6 x i32>, ptr addrspace(10) @wide_i32_6
  %c = icmp ne <6 x i32> %vc, zeroinitializer
  %r = select <6 x i1> %c, <6 x float> %va, <6 x float> %va
  store <6 x float> %r, ptr addrspace(10) @wide_f32_6
  ret void
}

define internal void @select_half9() {
entry:
  ; CHECK: OpFunction
  ; CHECK-COUNT-2: OpSelect %[[#vec4_float_16]]
  ; CHECK: OpSelect %[[#float_16]]
  ; CHECK: OpFunctionEnd
  %vc = load <9 x i32>, ptr addrspace(10) @wide_i32_9
  %c = icmp ne <9 x i32> %vc, zeroinitializer
  %vt = load <9 x half>, ptr addrspace(10) @wide_f16_9
  %vf = load <9 x half>, ptr addrspace(10) @wide_f16_9
  %r = select <9 x i1> %c, <9 x half> %vt, <9 x half> %vf
  store <9 x half> %r, ptr addrspace(10) @wide_f16_9
  ret void
}

define internal void @select_float16() {
entry:
  ; CHECK: OpFunction
  ; CHECK-COUNT-4: OpSelect %[[#vec4_float_32]]
  ; CHECK: OpFunctionEnd
  %vc = load <16 x i32>, ptr addrspace(10) @wide_i32_16
  %c = icmp ne <16 x i32> %vc, zeroinitializer
  %vt = load <16 x float>, ptr addrspace(10) @wide_f32_16
  %vf = load <16 x float>, ptr addrspace(10) @wide_f32_16
  %r = select <16 x i1> %c, <16 x float> %vt, <16 x float> %vf
  store <16 x float> %r, ptr addrspace(10) @wide_f32_16
  ret void
}

define internal void @select_int6_from_shuffle() {
entry:
  ; CHECK: OpFunction
  ; CHECK: %[[#shuffle_i32:]] = OpLoad %[[#vec4_int_32]]
  ; CHECK: OpCompositeExtract %[[#int_32]] %[[#shuffle_i32]] 0
  ; CHECK: OpSelect %[[#vec4_int_32]]
  ; CHECK: OpSelect %[[#vec2_int_32]]
  ; CHECK: OpFunctionEnd
  %vec = load <4 x i32>, ptr addrspace(10) @shuffle_i32_4
  %va = shufflevector <4 x i32> %vec, <4 x i32> %vec,
            <6 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5>
  %vc = load <6 x i32>, ptr addrspace(10) @wide_i32_6
  %c = icmp ne <6 x i32> %vc, zeroinitializer
  %r = select <6 x i1> %c, <6 x i32> %va, <6 x i32> %va
  store <6 x i32> %r, ptr addrspace(10) @wide_i32_6
  ret void
}

define internal void @select_short9() {
entry:
  ; CHECK: OpFunction
  ; CHECK-COUNT-2: OpSelect %[[#vec4_int_16]]
  ; CHECK: OpSelect %[[#int_16]]
  ; CHECK: OpFunctionEnd
  %vc = load <9 x i32>, ptr addrspace(10) @wide_i32_9
  %c = icmp ne <9 x i32> %vc, zeroinitializer
  %vt = load <9 x i16>, ptr addrspace(10) @wide_i16_9
  %vf = load <9 x i16>, ptr addrspace(10) @wide_i16_9
  %r = select <9 x i1> %c, <9 x i16> %vt, <9 x i16> %vf
  store <9 x i16> %r, ptr addrspace(10) @wide_i16_9
  ret void
}

define internal void @select_int16() {
entry:
  ; CHECK: OpFunction
  ; CHECK-COUNT-4: OpSelect %[[#vec4_int_32]]
  ; CHECK: OpFunctionEnd
  %vc = load <16 x i32>, ptr addrspace(10) @wide_i32_16
  %c = icmp ne <16 x i32> %vc, zeroinitializer
  %vt = load <16 x i32>, ptr addrspace(10) @wide_i32_16
  %vf = load <16 x i32>, ptr addrspace(10) @wide_i32_16
  %r = select <16 x i1> %c, <16 x i32> %vt, <16 x i32> %vf
  store <16 x i32> %r, ptr addrspace(10) @wide_i32_16
  ret void
}

define void @main() #0 {
entry:
  call void @select_float6_from_shuffle()
  call void @select_half9()
  call void @select_float16()
  call void @select_int6_from_shuffle()
  call void @select_short9()
  call void @select_int16()
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-NOT: OpCapability Vector16
; CHECK-DAG: OpCapability Float16
; CHECK-DAG: %[[#ext:]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG: %[[#void:]] = OpTypeVoid
; CHECK-DAG: %[[#f32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#vec4f32:]] = OpTypeVector %[[#f32]] 4
; CHECK-DAG: %[[#vec2f32:]] = OpTypeVector %[[#f32]] 2
; CHECK-DAG: %[[#f16:]] = OpTypeFloat 16
; CHECK-DAG: %[[#vec4f16:]] = OpTypeVector %[[#f16]] 4

@wide_f32_6 = internal addrspace(10) global [6 x float] zeroinitializer
@wide_f16_9 = internal addrspace(10) global [9 x half] zeroinitializer
@wide_f32_16 = internal addrspace(10) global [16 x float] zeroinitializer
@shuffle_f32_4 = internal addrspace(10) global <4 x float> zeroinitializer

define internal void @tan_float6_from_shuffle() {
entry:
  ; CHECK-LABEL: %{{[0-9]+}} = OpFunction %{{[0-9]+}} None %{{[0-9]+}} ; -- Begin function tan_float6_from_shuffle
  ; CHECK: %{{[0-9]+}} = OpLoad %[[#vec4f32]]
  ; CHECK: %{{[0-9]+}} = OpExtInst %[[#vec4f32]] %[[#ext]] Tan
  ; CHECK: %{{[0-9]+}} = OpExtInst %[[#vec2f32]] %[[#ext]] Tan
  %vec = load <4 x float>, ptr addrspace(10) @shuffle_f32_4
  %va = shufflevector <4 x float> %vec, <4 x float> %vec,
            <6 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5>
  %r = call <6 x float> @llvm.tan.v6f32(<6 x float> %va)
  store <6 x float> %r, ptr addrspace(10) @wide_f32_6
  ret void
}

define internal void @tan_half9() {
entry:
  ; CHECK-LABEL: %{{[0-9]+}} = OpFunction %{{[0-9]+}} None %{{[0-9]+}} ; -- Begin function tan_half9
  ; CHECK-COUNT-2: %{{[0-9]+}} = OpExtInst %[[#vec4f16]] %[[#ext]] Tan
  ; CHECK: %{{[0-9]+}} = OpExtInst %[[#f16]] %[[#ext]] Tan
  %va = load <9 x half>, ptr addrspace(10) @wide_f16_9
  %r = call <9 x half> @llvm.tan.v9f16(<9 x half> %va)
  store <9 x half> %r, ptr addrspace(10) @wide_f16_9
  ret void
}

define internal void @tan_float16() {
entry:
  ; CHECK-LABEL: %{{[0-9]+}} = OpFunction %{{[0-9]+}} None %{{[0-9]+}} ; -- Begin function tan_float16
  ; CHECK-COUNT-4: %{{[0-9]+}} = OpExtInst %[[#vec4f32]] %[[#ext]] Tan
  %va = load <16 x float>, ptr addrspace(10) @wide_f32_16
  %r = call <16 x float> @llvm.tan.v16f32(<16 x float> %va)
  store <16 x float> %r, ptr addrspace(10) @wide_f32_16
  ret void
}

define void @main() #0 {
entry:
  call void @tan_float6_from_shuffle()
  call void @tan_half9()
  call void @tan_float16()
  ret void
}

declare <6 x float> @llvm.tan.v6f32(<6 x float>)
declare <9 x half> @llvm.tan.v9f16(<9 x half>)
declare <16 x float> @llvm.tan.v16f32(<16 x float>)

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

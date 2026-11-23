; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-NOT: OpCapability Vector16
; CHECK-DAG: OpCapability Float16
; CHECK-DAG: OpCapability Int16
; CHECK-DAG: %[[#ext:]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG: %[[#void:]] = OpTypeVoid
; CHECK-DAG: %[[#f32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#vec4f32:]] = OpTypeVector %[[#f32]] 4
; CHECK-DAG: %[[#vec2f32:]] = OpTypeVector %[[#f32]] 2
; CHECK-DAG: %[[#f16:]] = OpTypeFloat 16
; CHECK-DAG: %[[#vec4f16:]] = OpTypeVector %[[#f16]] 4
; CHECK-DAG: %[[#i32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#vec4i32:]] = OpTypeVector %[[#i32]] 4
; CHECK-DAG: %[[#vec2i32:]] = OpTypeVector %[[#i32]] 2
; CHECK-DAG: %[[#i16:]] = OpTypeInt 16 0
; CHECK-DAG: %[[#vec4i16:]] = OpTypeVector %[[#i16]] 4

@shuffle_f32_4 = internal addrspace(10) global <4 x float> zeroinitializer
@wide_f32_6 = internal addrspace(10) global [6 x float] zeroinitializer
@wide_f16_9 = internal addrspace(10) global [9 x half] zeroinitializer
@wide_f32_16 = internal addrspace(10) global [16 x float] zeroinitializer
@shuffle_i32_4 = internal addrspace(10) global <4 x i32> zeroinitializer
@wide_i32_6 = internal addrspace(10) global [6 x i32] zeroinitializer
@wide_i16_9 = internal addrspace(10) global [9 x i16] zeroinitializer
@wide_i32_16 = internal addrspace(10) global [16 x i32] zeroinitializer

define internal void @abs_float6_from_shuffle() {
entry:
  ; CHECK: OpFunction %[[#void]] None
  ; CHECK: OpExtInst %[[#vec4f32]] %[[#ext]] FAbs
  ; CHECK: OpExtInst %[[#vec2f32]] %[[#ext]] FAbs
  %vec = load <4 x float>, ptr addrspace(10) @shuffle_f32_4
  %va = shufflevector <4 x float> %vec, <4 x float> %vec,
            <6 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5>
  %r = call <6 x float> @llvm.fabs.v6f32(<6 x float> %va)
  store <6 x float> %r, ptr addrspace(10) @wide_f32_6
  ret void
}

define internal void @abs_half9() {
entry:
  ; CHECK: OpFunction %[[#void]] None
  ; CHECK-COUNT-2: OpExtInst %[[#vec4f16]] %[[#ext]] FAbs
  ; CHECK: OpExtInst %[[#f16]] %[[#ext]] FAbs
  %va = load <9 x half>, ptr addrspace(10) @wide_f16_9
  %r = call <9 x half> @llvm.fabs.v9f16(<9 x half> %va)
  store <9 x half> %r, ptr addrspace(10) @wide_f16_9
  ret void
}

define internal void @abs_float16() {
entry:
  ; CHECK: OpFunction %[[#void]] None
  ; CHECK-COUNT-4: OpExtInst %[[#vec4f32]] %[[#ext]] FAbs
  %va = load <16 x float>, ptr addrspace(10) @wide_f32_16
  %r = call <16 x float> @llvm.fabs.v16f32(<16 x float> %va)
  store <16 x float> %r, ptr addrspace(10) @wide_f32_16
  ret void
}

define internal void @abs_int6_from_shuffle() {
entry:
  ; CHECK: OpFunction %[[#void]] None
  ; CHECK: OpExtInst %[[#vec4i32]] %[[#ext]] SAbs
  ; CHECK: OpExtInst %[[#vec2i32]] %[[#ext]] SAbs
  %vec = load <4 x i32>, ptr addrspace(10) @shuffle_i32_4
  %va = shufflevector <4 x i32> %vec, <4 x i32> %vec,
            <6 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5>
  %r = call <6 x i32> @llvm.abs.v6i32(<6 x i32> %va, i1 false)
  store <6 x i32> %r, ptr addrspace(10) @wide_i32_6
  ret void
}

define internal void @abs_short9() {
entry:
  ; CHECK: OpFunction %[[#void]] None
  ; CHECK-COUNT-2: OpExtInst %[[#vec4i16]] %[[#ext]] SAbs
  ; CHECK: OpExtInst %[[#i16]] %[[#ext]] SAbs
  %va = load <9 x i16>, ptr addrspace(10) @wide_i16_9
  %r = call <9 x i16> @llvm.abs.v9i16(<9 x i16> %va, i1 false)
  store <9 x i16> %r, ptr addrspace(10) @wide_i16_9
  ret void
}

define internal void @abs_int16() {
entry:
  ; CHECK: OpFunction %[[#void]] None
  ; CHECK-COUNT-4: OpExtInst %[[#vec4i32]] %[[#ext]] SAbs
  %va = load <16 x i32>, ptr addrspace(10) @wide_i32_16
  %r = call <16 x i32> @llvm.abs.v16i32(<16 x i32> %va, i1 false)
  store <16 x i32> %r, ptr addrspace(10) @wide_i32_16
  ret void
}

define void @main() #0 {
entry:
  call void @abs_float6_from_shuffle()
  call void @abs_half9()
  call void @abs_float16()
  call void @abs_int6_from_shuffle()
  call void @abs_short9()
  call void @abs_int16()
  ret void
}

declare <6 x float> @llvm.fabs.v6f32(<6 x float>)
declare <9 x half> @llvm.fabs.v9f16(<9 x half>)
declare <16 x float> @llvm.fabs.v16f32(<16 x float>)
declare <6 x i32> @llvm.abs.v6i32(<6 x i32>, i1)
declare <9 x i16> @llvm.abs.v9i16(<9 x i16>, i1)
declare <16 x i32> @llvm.abs.v16i32(<16 x i32>, i1)

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

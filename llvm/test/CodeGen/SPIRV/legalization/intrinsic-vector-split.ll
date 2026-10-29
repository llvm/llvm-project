; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-DAG: %[[#glsl:]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG: %[[#uint:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#f32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#bool:]] = OpTypeBool
; CHECK-DAG: %[[#v2f32:]] = OpTypeVector %[[#f32]] 2
; CHECK-DAG: %[[#v4f32:]] = OpTypeVector %[[#f32]] 4
; CHECK-DAG: %[[#v2bool:]] = OpTypeVector %[[#bool]] 2
; CHECK-DAG: %[[#v4bool:]] = OpTypeVector %[[#bool]] 4
; CHECK-DAG: %[[#scope:]] = OpConstant %[[#uint]] 3

@f = internal addrspace(10) global [6 x float] zeroinitializer
@i = internal addrspace(10) global [6 x i32] zeroinitializer

define void @main() #0 {
entry:
  %v = load <6 x float>, ptr addrspace(10) @f

; The scalar lane index is reused by every part.
; CHECK: %[[#rl0:]] = OpGroupNonUniformShuffle %[[#v4f32]] %[[#scope]] %[[#]] %[[#scope]]
; CHECK: %[[#rl1:]] = OpGroupNonUniformShuffle %[[#v2f32]] %[[#scope]] %[[#]] %[[#scope]]
  %rl = call <6 x float> @llvm.spv.wave.readlane.v6f32(<6 x float> %v, i32 3)

; CHECK: %[[#cl0:]] = OpExtInst %[[#v4f32]] %[[#glsl]] NClamp %[[#rl0]] %[[#]] %[[#]]
; CHECK: %[[#cl1:]] = OpExtInst %[[#v2f32]] %[[#glsl]] NClamp %[[#rl1]] %[[#]] %[[#]]
  %cl = call <6 x float> @llvm.spv.nclamp.v6f32(<6 x float> %rl, <6 x float> %v, <6 x float> %v)
  store <6 x float> %cl, ptr addrspace(10) @f

; CHECK: OpIsInf %[[#v4bool]] %[[#cl0]]
; CHECK: OpIsInf %[[#v2bool]] %[[#cl1]]
  %inf = call <6 x i1> @llvm.spv.isinf.v6f32(<6 x float> %cl)
  %ext = zext <6 x i1> %inf to <6 x i32>
  store <6 x i32> %ext, ptr addrspace(10) @i

; A single trailing element becomes a scalar call.
; CHECK: OpExtInst %[[#v4f32]] %[[#glsl]] Fract
; CHECK: OpExtInst %[[#f32]] %[[#glsl]] Fract
  %v5 = load <5 x float>, ptr addrspace(10) @f
  %fr = call <5 x float> @llvm.spv.frac.v5f32(<5 x float> %v5)
  store <5 x float> %fr, ptr addrspace(10) @f
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

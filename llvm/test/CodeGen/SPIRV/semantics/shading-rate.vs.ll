; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan1.3-vertex -spirv-ext=+SPV_KHR_fragment_shading_rate %s -o - | FileCheck %s
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan1.3-vertex -spirv-ext=+SPV_KHR_fragment_shading_rate %s -o - | FileCheck %s --check-prefix=NOFLAT
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-vertex -spirv-ext=+SPV_KHR_fragment_shading_rate %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; Simplified LLVM IR of a vertex shader writing SV_ShadingRate.

; The output is a per-primitive built-in and must not be decorated Flat. 
; The NOFLAT prefix has only a NOT directive, so it scans the whole output.
; NOFLAT-NOT: Flat

; CHECK-DAG: OpCapability FragmentShadingRateKHR
; CHECK-DAG: OpExtension "SPV_KHR_fragment_shading_rate"
; CHECK-DAG: OpDecorate %[[#POS:]] BuiltIn Position
; CHECK-DAG: OpDecorate %[[#RATE:]] BuiltIn PrimitiveShadingRateKHR
; CHECK-DAG: %[[#int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#ptr_o:]] = OpTypePointer Output %[[#int]]
; CHECK-DAG: %[[#RATE]] = OpVariable %[[#ptr_o]] Output

@SV_Position = external hidden thread_local addrspace(8) global <4 x float>, !spirv.Decorations !0
@SV_ShadingRate = external hidden thread_local addrspace(8) global i32, !spirv.Decorations !3

define void @main() #0 {
entry:
  store <4 x float> <float 0.0, float 0.0, float 0.0, float 1.0>, ptr addrspace(8) @SV_Position, align 16
  store i32 5, ptr addrspace(8) @SV_ShadingRate, align 4
  ret void
}

attributes #0 = { "hlsl.shader"="vertex" }

; SV_Position output: BuiltIn Position (0).
!0 = !{!1}
!1 = !{i32 11, i32 0}
; SV_ShadingRate output: BuiltIn PrimitiveShadingRateKHR (4432), no Flat.
!3 = !{!4}
!4 = !{i32 11, i32 4432}

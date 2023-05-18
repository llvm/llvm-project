; RUN: %if directx-registered-target %{ opt -S -dxil-metadata-emit < %s | FileCheck %s --check-prefix=DXIL-CHECK %}
; RUN: %if spirv-registered-target   %{ llc %s -mtriple=spirv-unknown-unknown -o - | FileCheck %s --check-prefix=SPIRV-CHECK %}

target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-unknown-shadermodel6.7-compute"

;DXIL-CHECK:!dx.entryPoints = !{![[entry:[0-9]+]]}

;DXIL-CHECK:![[entry]] = !{ptr @entry, !"entry", null, null, ![[extra:[0-9]+]]}
;DXIL-CHECK:![[extra]] = !{i32 4, ![[numthreads:[0-9]+]]}
;DXIL-CHECK:![[numthreads]] = !{i32 1, i32 2, i32 1}

;SPIRV-CHECK:                     OpCapability Shader
;SPIRV-CHECK:                     OpMemoryModel Logical GLSL450
;SPIRV-CHECK:                     OpEntryPoint GLCompute [[main:%[0-9]+]] "entry"
;SPIRV-CHECK:                     OpExecutionMode [[main]] LocalSize 1 2 1
;SPIRV-CHECK:  [[void:%[0-9]+]] = OpTypeVoid
;SPIRV-CHECK: [[ftype:%[0-9]+]] = OpTypeFunction [[void]]
;SPIRV-CHECK:          [[main]] = OpFunction [[void]] DontInline [[ftype]]

; Function Attrs: noinline nounwind
define void @entry() #0 {
entry:
  ret void
}

attributes #0 = { noinline nounwind "hlsl.numthreads"="1,2,1" "hlsl.shader"="compute" }

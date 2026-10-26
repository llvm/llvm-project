; Constrained fadd/fsub do not disable contraction (translator parity).

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpEntryPoint Kernel %[[#]] "k"
; CHECK-NOT: ContractionOff
; CHECK: OpFAdd
; CHECK: OpFSub

define spir_kernel void @k(float %a, float %b, ptr addrspace(1) %p) #0 {
entry:
  %add = call float @llvm.experimental.constrained.fadd.f32(float %a, float %b, metadata !"round.dynamic", metadata !"fpexcept.strict") #0
  %sub = call float @llvm.experimental.constrained.fsub.f32(float %add, float %b, metadata !"round.dynamic", metadata !"fpexcept.strict") #0
  store float %sub, ptr addrspace(1) %p
  ret void
}

declare float @llvm.experimental.constrained.fadd.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.fsub.f32(float, float, metadata, metadata)

attributes #0 = { strictfp }

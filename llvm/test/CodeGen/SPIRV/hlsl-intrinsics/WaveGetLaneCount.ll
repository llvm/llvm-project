; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-vulkan-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-unknown %s -o - -filetype=obj | spirv-val %}

target triple = "spirv-unknown-vulkan-compute"

; CHECK-DAG: OpCapability Shader
; CHECK-DAG: OpCapability GroupNonUniform
; CHECK-DAG: OpDecorate %[[#var:]] BuiltIn SubgroupSize
; CHECK-DAG: %[[#uint:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#ptr:]] = OpTypePointer Input %[[#uint]]
; CHECK-DAG: %[[#var]] = OpVariable %[[#ptr]] Input

; CHECK-NOT: OpDecorate %[[#var]] LinkageAttributes

define spir_func i32 @test_fun() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
; CHECK: %[[#count:]] = OpLoad %[[#uint]] %[[#var]]
  %1 = call i32 @llvm.spv.wave.get.lane.count()
      [ "convergencectrl"(token %0) ]
; CHECK: OpReturnValue %[[#count]]
  ret i32 %1
}

declare i32 @llvm.spv.wave.get.lane.count() #0

declare token @llvm.experimental.convergence.entry()

attributes #0 = { convergent }

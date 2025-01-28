; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#bool:]] = OpTypeBool
; CHECK-DAG: %[[#uint:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#scope:]] = OpConstant %[[#uint]] 3
; CHECK-DAG: OpCapability GroupNonUniformVote

; CHECK-LABEL: Begin function test_wave_all
define i1 @test_wave_all(i1 %p1) #0 {
entry:
; CHECK: %[[#param:]] = OpFunctionParameter %[[#bool]]
; CHECK: %{{.+}} = OpGroupNonUniformAll %[[#bool]] %[[#scope]] %[[#param]]
  %0 = call token @llvm.experimental.convergence.entry()
  %ret = call i1 @llvm.spv.wave.all(i1 %p1) [ "convergencectrl"(token %0) ]
  ret i1 %ret
}

declare i1 @llvm.spv.wave.all(i1) #0

attributes #0 = { convergent }

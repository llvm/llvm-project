; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#bool:]] = OpTypeBool
; CHECK-DAG: %[[#uint:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#scope:]] = OpConstant %[[#uint]] 3
; CHECK-DAG: %[[#bitmask:]] = OpTypeVector %[[#uint]] 4
; CHECK-DAG: OpCapability GroupNonUniformBallot

; CHECK-LABEL: Begin function test_wave_ballot
define <4 x i32> @test_wave_ballot(i1 %p1) #0 {
entry:
; CHECK: %[[#param:]] = OpFunctionParameter %[[#bool]]
; CHECK: %{{.+}} = OpGroupNonUniformBallot %[[#bitmask]] %[[#scope]] %[[#param]]
  %0 = call token @llvm.experimental.convergence.entry()
  %ret = call <4 x i32> @llvm.spv.subgroup.ballot(i1 %p1) [ "convergencectrl"(token %0) ]
  ret <4 x i32> %ret
}

declare <4 x i32> @llvm.spv.subgroup.ballot(i1) #0

attributes #0 = { convergent }

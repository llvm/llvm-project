; RUN: llc -O0 -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#ScopeCrossDevice:]] = OpConstantNull %[[#Int]]
; CHECK-DAG: %[[#ScopeWorkgroup:]] = OpConstant %[[#Int]] 2{{$}}
; CHECK-DAG: %[[#Acquire:]] = OpConstant %[[#Int]] 2370
; CHECK-DAG: %[[#Release:]] = OpConstant %[[#Int]] 2372
; CHECK-DAG: %[[#AcqRel:]] = OpConstant %[[#Int]] 2376
; CHECK-DAG: %[[#SeqCst:]] = OpConstant %[[#Int]] 2384

; CHECK: OpMemoryBarrier %[[#ScopeCrossDevice]] %[[#Acquire]]
define void @fence_acquire() {
  fence acquire
  ret void
}

; CHECK: OpMemoryBarrier %[[#ScopeCrossDevice]] %[[#Release]]
define void @fence_release() {
  fence release
  ret void
}

; CHECK: OpMemoryBarrier %[[#ScopeCrossDevice]] %[[#AcqRel]]
define void @fence_acq_rel() {
  fence acq_rel
  ret void
}

; CHECK: OpMemoryBarrier %[[#ScopeWorkgroup]] %[[#SeqCst]]
define void @fence_workgroup_seq_cst() {
  fence syncscope("workgroup") seq_cst
  ret void
}

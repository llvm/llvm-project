; RUN: llc -O0 -mtriple=spirv64-unknown-unknown < %s -o - -filetype=asm | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown < %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv64-vulkan-unknown < %s -o - -filetype=asm | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-vulkan-unknown < %s -o - -filetype=obj | spirv-val %}

; CHECK: OpName %[[#Global:]] "global_var"
; CHECK: OpDecorate %[[#Global]] LinkageAttributes "global_var" Export
; CHECK: %[[#Int32Ty:]] = OpTypeInt 32 0
; CHECK: %[[#Int32PtrTy:]] = OpTypePointer CrossWorkgroup %[[#Int32Ty]]
; CHECK: %[[#Initializer:]] = OpConstant{{.*}} %[[#Int32Ty]]
; CHECK: OpVariable %[[#Int32PtrTy:]] CrossWorkgroup %[[#Initializer]]

; Verify we emit global definitions even if there are no functions.

@global_var = addrspace(1) global i32 zeroinitializer

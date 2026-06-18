; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Chained extractvalue from an aggregate-returning call used to crash in
; foldImm because the outer extractvalue was left as raw IR over a
; multi-register spv_extractv result.

%ty = type { ptr addrspace(4), ptr addrspace(4), [8 x i8] }

declare %ty @arr_ret()

; CHECK: OpFunction
; CHECK: %[[#AGG:]] = OpFunctionCall %[[#]] %[[#]]
; CHECK: %[[#ARR:]] = OpCompositeExtract %[[#]] %[[#AGG]] 2
; CHECK: %[[#ELT:]] = OpCompositeExtract %[[#]] %[[#ARR]] 0
; CHECK: OpStore %[[#]] %[[#ELT]]
define void @chain(ptr addrspace(4) %p) {
  %1 = call %ty @arr_ret()
  %2 = extractvalue %ty %1, 2
  %3 = extractvalue [8 x i8] %2, 0
  store i8 %3, ptr addrspace(4) %p, align 1
  ret void
}

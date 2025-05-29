; RUN: llc -verify-machineinstrs -O3 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O3 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG:                     %[[#uint:]] = OpTypeInt 32 0
; CHECK-DAG:                   %[[#uint_0:]] = OpConstant %[[#uint]] 0
; CHECK-DAG:                 %[[#ptr_uint:]] = OpTypePointer Private %[[#uint]]
; CHECK-DAG:                      %[[#var:]] = OpVariable %[[#ptr_uint]] Private %[[#uint_0]]

; CHECK-DAG:  OpName %[[#func_simple:]] "simple"
; CHECK-DAG:  OpName %[[#func_chain:]] "chain"

@global = internal addrspace(10) global i32 zeroinitializer

define void @simple() {
; CHECK: %[[#func_simple]] = OpFunction
entry:
  %ptr = getelementptr i32, ptr addrspace(10) @global, i32 0
  %casted = addrspacecast ptr addrspace(10) %ptr to ptr
  %val = load i32, ptr %casted
; CHECK: %{{.*}} = OpLoad %[[#uint]] %[[#var]] Aligned 4
  ret void
}

define void @chain() {
; CHECK: %[[#func_chain]] = OpFunction
entry:
  %a = getelementptr i32, ptr addrspace(10) @global, i32 0
  %b = addrspacecast ptr addrspace(10) %a to ptr
  %c = getelementptr i32, ptr %b, i32 0
  %d = addrspacecast ptr %c to ptr addrspace(10)
  %e = addrspacecast ptr addrspace(10) %d to ptr

  %val = load i32, ptr %e
; CHECK: %{{.*}} = OpLoad %[[#uint]] %[[#var]] Aligned 4
  ret void
}

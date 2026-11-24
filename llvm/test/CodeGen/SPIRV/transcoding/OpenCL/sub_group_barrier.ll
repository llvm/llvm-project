; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV --implicit-check-not=OpFunctionCall
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32v1.2-unknown-unknown %s -o - -filetype=obj | spirv-val --target-env opencl2.2 %}
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV --implicit-check-not=OpFunctionCall
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64v1.2-unknown-unknown %s -o - -filetype=obj | spirv-val --target-env opencl2.2 %}

; The one-argument form defaults to Subgroup execution and memory scopes.
; Explicit memory scopes do not change the Subgroup execution scope,
; regardless of whether CLK_LOCAL_MEM_FENCE is set.
; https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_C.html

; CHECK-SPIRV-DAG: OpName %[[#TEST_CONST_FLAGS:]] "test_barrier_const_flags"
; CHECK-SPIRV: %[[#UINT:]] = OpTypeInt 32 0

;; 0x10 SequentiallyConsistent, with no memory-class bits for zero flags
; CHECK-SPIRV-DAG: %[[#SEQ_CST:]] = OpConstant %[[#UINT]] 16{{$}}
;; 0x10 SequentiallyConsistent + 0x100 WorkgroupMemory
; CHECK-SPIRV-DAG: %[[#LOCAL:]] = OpConstant %[[#UINT]] 272{{$}}
;; 0x10 SequentiallyConsistent + 0x200 CrossWorkgroupMemory
; CHECK-SPIRV-DAG: %[[#GLOBAL:]] = OpConstant %[[#UINT]] 528{{$}}
;; 0x10 SequentiallyConsistent + 0x800 ImageMemory
; CHECK-SPIRV-DAG: %[[#IMAGE:]] = OpConstant %[[#UINT]] 2064{{$}}
;; 0x10 SequentiallyConsistent + 0x100 WorkgroupMemory + 0x200 CrossWorkgroupMemory
; CHECK-SPIRV-DAG: %[[#LOCAL_GLOBAL:]] = OpConstant %[[#UINT]] 784{{$}}
;; 0x10 SequentiallyConsistent + 0x100 WorkgroupMemory + 0x800 ImageMemory
; CHECK-SPIRV-DAG: %[[#LOCAL_IMAGE:]] = OpConstant %[[#UINT]] 2320{{$}}
;; 0x10 SequentiallyConsistent + 0x200 CrossWorkgroupMemory + 0x800 ImageMemory
; CHECK-SPIRV-DAG: %[[#GLOBAL_IMAGE:]] = OpConstant %[[#UINT]] 2576{{$}}
;; 0x10 SequentiallyConsistent + 0x100 WorkgroupMemory + 0x200 CrossWorkgroupMemory + 0x800 ImageMemory
; CHECK-SPIRV-DAG: %[[#LOCAL_GLOBAL_IMAGE:]] = OpConstant %[[#UINT]] 2832{{$}}

;; Scopes:
;; 3 Subgroup
; CHECK-SPIRV-DAG: %[[#SCOPE_SUBGROUP:]] = OpConstant %[[#UINT]] 3{{$}}
;; 2 Workgroup
; CHECK-SPIRV-DAG: %[[#SCOPE_WORK_GROUP:]] = OpConstant %[[#UINT]] 2{{$}}
;; 4 Invocation
; CHECK-SPIRV-DAG: %[[#SCOPE_INVOCATION:]] = OpConstant %[[#UINT]] 4{{$}}
;; 1 Device
; CHECK-SPIRV-DAG: %[[#SCOPE_DEVICE:]] = OpConstant %[[#UINT]] 1{{$}}
;; 0 CrossDevice
; CHECK-SPIRV-DAG: %[[#SCOPE_CROSS_DEVICE:]] = OpConstantNull %[[#UINT]]

; CHECK-SPIRV: %[[#TEST_CONST_FLAGS]] = OpFunction %[[#]]
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_SUBGROUP]] %[[#SCOPE_SUBGROUP]] %[[#LOCAL]]
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_SUBGROUP]] %[[#SCOPE_SUBGROUP]] %[[#GLOBAL]]
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_SUBGROUP]] %[[#SCOPE_SUBGROUP]] %[[#IMAGE]]
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_SUBGROUP]] %[[#SCOPE_SUBGROUP]] %[[#SEQ_CST]]
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_SUBGROUP]] %[[#SCOPE_SUBGROUP]] %[[#LOCAL_GLOBAL]]
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_SUBGROUP]] %[[#SCOPE_SUBGROUP]] %[[#LOCAL_IMAGE]]
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_SUBGROUP]] %[[#SCOPE_SUBGROUP]] %[[#GLOBAL_IMAGE]]
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_SUBGROUP]] %[[#SCOPE_SUBGROUP]] %[[#LOCAL_GLOBAL_IMAGE]]
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_SUBGROUP]] %[[#SCOPE_WORK_GROUP]] %[[#LOCAL]]
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_SUBGROUP]] %[[#SCOPE_SUBGROUP]] %[[#LOCAL]]
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_SUBGROUP]] %[[#SCOPE_SUBGROUP]] %[[#SEQ_CST]]
; Local bit clear: execution scope must remain Subgroup.
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_SUBGROUP]] %[[#SCOPE_INVOCATION]] %[[#GLOBAL]]
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_SUBGROUP]] %[[#SCOPE_WORK_GROUP]] %[[#GLOBAL]]
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_SUBGROUP]] %[[#SCOPE_DEVICE]] %[[#GLOBAL]]
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_SUBGROUP]] %[[#SCOPE_CROSS_DEVICE]] %[[#GLOBAL]]
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_SUBGROUP]] %[[#SCOPE_DEVICE]] %[[#IMAGE]]
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_SUBGROUP]] %[[#SCOPE_WORK_GROUP]] %[[#SEQ_CST]]
; Local bit set: memory scope must follow the argument.
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_SUBGROUP]] %[[#SCOPE_INVOCATION]] %[[#LOCAL]]
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_SUBGROUP]] %[[#SCOPE_DEVICE]] %[[#LOCAL]]
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_SUBGROUP]] %[[#SCOPE_CROSS_DEVICE]] %[[#LOCAL]]
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_SUBGROUP]] %[[#SCOPE_DEVICE]] %[[#LOCAL_GLOBAL]]
; CHECK-SPIRV: OpFunctionEnd

define spir_kernel void @test_barrier_const_flags() {
entry:
  call spir_func void @_Z17sub_group_barrierj(i32 1)
  call spir_func void @_Z17sub_group_barrierj(i32 2)
  call spir_func void @_Z17sub_group_barrierj(i32 4)
  call spir_func void @_Z17sub_group_barrierj(i32 0)
  call spir_func void @_Z17sub_group_barrierj(i32 3)
  call spir_func void @_Z17sub_group_barrierj(i32 5)
  call spir_func void @_Z17sub_group_barrierj(i32 6)
  call spir_func void @_Z17sub_group_barrierj(i32 7)
  call spir_func void @_Z17sub_group_barrierj12memory_scope(i32 1, i32 1)
  call spir_func void @_Z17sub_group_barrierj12memory_scope(i32 1, i32 4)
  call spir_func void @_Z17sub_group_barrierj12memory_scope(i32 0, i32 4)

  ; Local bit clear: global, image, and zero flags with explicit memory scopes.
  call spir_func void @_Z17sub_group_barrierj12memory_scope(i32 2, i32 0)
  call spir_func void @_Z17sub_group_barrierj12memory_scope(i32 2, i32 1)
  call spir_func void @_Z17sub_group_barrierj12memory_scope(i32 2, i32 2)
  call spir_func void @_Z17sub_group_barrierj12memory_scope(i32 2, i32 3)
  call spir_func void @_Z17sub_group_barrierj12memory_scope(i32 4, i32 2)
  call spir_func void @_Z17sub_group_barrierj12memory_scope(i32 0, i32 1)

  ; Local bit set: local and local+global flags with explicit memory scopes.
  call spir_func void @_Z17sub_group_barrierj12memory_scope(i32 1, i32 0)
  call spir_func void @_Z17sub_group_barrierj12memory_scope(i32 1, i32 2)
  call spir_func void @_Z17sub_group_barrierj12memory_scope(i32 1, i32 3)
  call spir_func void @_Z17sub_group_barrierj12memory_scope(i32 3, i32 2)
  ret void
}

declare spir_func void @_Z17sub_group_barrierj(i32)

declare spir_func void @_Z17sub_group_barrierj12memory_scope(i32, i32)

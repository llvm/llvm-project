; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Omitted order/scope arguments default to sequentially consistent/device.
; A local pointer contributes WorkgroupMemory, independently of the scope.
; Explicit order and scope arguments must override the defaults.
; CHECK-DAG: %[[#INT:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#PTR:]] = OpTypePointer Workgroup %[[#INT]]
; CHECK-DAG: %[[#DEVICE:]] = OpConstant %[[#INT]] 1{{$}}
; CHECK-DAG: %[[#WORKGROUP:]] = OpConstant %[[#INT]] 2{{$}}
; SequentiallyConsistent | WorkgroupMemory = 0x110.
; CHECK-DAG: %[[#SEQCST_LOCAL:]] = OpConstant %[[#INT]] 272{{$}}
; AcquireRelease | WorkgroupMemory = 0x108.
; CHECK-DAG: %[[#ACQREL_LOCAL:]] = OpConstant %[[#INT]] 264{{$}}
; Acquire | WorkgroupMemory = 0x102.
; CHECK-DAG: %[[#ACQUIRE_LOCAL:]] = OpConstant %[[#INT]] 258{{$}}

; CHECK: OpFunction
; CHECK-NEXT: %[[#P:]] = OpFunctionParameter %[[#PTR]]
; CHECK-NEXT: %[[#V:]] = OpFunctionParameter %[[#INT]]
; CHECK: OpAtomicIAdd %[[#INT]] %[[#P]] %[[#DEVICE]] %[[#SEQCST_LOCAL]] %[[#V]]
; CHECK: OpAtomicIAdd %[[#INT]] %[[#P]] %[[#DEVICE]] %[[#ACQREL_LOCAL]] %[[#V]]
; CHECK: OpAtomicIAdd %[[#INT]] %[[#P]] %[[#WORKGROUP]] %[[#ACQUIRE_LOCAL]] %[[#V]]
; CHECK: OpReturn
; CHECK-NEXT: OpFunctionEnd
define spir_func void @test_atomic_fetch_add_local(ptr addrspace(3) %p, i32 %v) {
  %default = call spir_func i32 @_Z16atomic_fetch_addPU3AS3VU7_Atomicii(ptr addrspace(3) %p, i32 %v)
  %order = call spir_func i32 @_Z25atomic_fetch_add_explicitPU3AS3VU7_Atomicii12memory_order(ptr addrspace(3) %p, i32 %v, i32 4)
  %scope = call spir_func i32 @_Z25atomic_fetch_add_explicitPU3AS3VU7_Atomicii12memory_order12memory_scope(ptr addrspace(3) %p, i32 %v, i32 2, i32 1)
  ret void
}

declare spir_func i32 @_Z16atomic_fetch_addPU3AS3VU7_Atomicii(ptr addrspace(3), i32)
declare spir_func i32 @_Z25atomic_fetch_add_explicitPU3AS3VU7_Atomicii12memory_order(ptr addrspace(3), i32, i32)
declare spir_func i32 @_Z25atomic_fetch_add_explicitPU3AS3VU7_Atomicii12memory_order12memory_scope(ptr addrspace(3), i32, i32, i32)

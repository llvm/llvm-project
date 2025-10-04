; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_predicated_io %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s --spirv-ext=+SPV_INTEL_predicated_io -o - -filetype=obj | spirv-val %}

; CHECK-ERROR: LLVM ERROR: OpPredicated[Load/Store]INTEL
; CHECK-ERROR-SAME: instructions require the following SPIR-V extension: SPV_INTEL_predicated_io

; CHECK-DAG: Capability PredicatedIOINTEL
; CHECK-DAG: Extension "SPV_INTEL_predicated_io"

; CHECK-DAG: %[[Int32Ty:[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: %[[Const0:[0-9]+]] = OpConstant %[[Int32Ty]] 0
; CHECK-DAG: %[[VoidTy:[0-9]+]] = OpTypeVoid
; CHECK-DAG: %[[IntPtrTy:[0-9]+]] = OpTypePointer CrossWorkgroup %[[Int32Ty]]
; CHECK-DAG: %[[BoolTy:[0-9]+]] = OpTypeBool
; CHECK: %[[LoadPtr:]] = FunctionParameter %[[IntPtrTy]]
; CHECK: %[[StorePtr:]] = FunctionParameter %[[IntPtrTy]]
; CHECK: %[[DefaultVal:]] = FunctionParameter %[[Int32Ty]]
; CHECK: %[[StoreObj:]] = FunctionParameter %[[Int32Ty]]
; CHECK: %[[Predicate:]] = FunctionParameter %[[BoolTy]]
; CHECK: PredicatedLoadINTEL %[[Int32Ty]] %[[Result1:]] %[[LoadPtr]] %[[Predicate]] %[[DefaultVal]]
; CHECK: PredicatedLoadINTEL %[[Int32Ty]] %[[Result2:]] %[[LoadPtr]] %[[Predicate]] %[[DefaultVal]] %[[Const0]]
; CHECK: PredicatedStoreINTEL %[[StorePtr]] %[[StoreObj]] %[[Predicate]]
; CHECK: PredicatedStoreINTEL %[[StorePtr]] %[[StoreObj]] %[[Predicate]] %[[Const0]]

define spir_func void @foo(ptr addrspace(1) %load_pointer, ptr addrspace(1) %store_pointer, i32  %default_value, i32 %store_object, i1 zeroext %predicate) {
entry:
  %1 = call spir_func i32 @_Z27__spirv_PredicatedLoadINTELPU3AS1Kibi(ptr addrspace(1) %load_pointer, i1 %predicate, i32 %default_value)
  %2 = call spir_func i32 @_Z27__spirv_PredicatedLoadINTELPU3AS1Kibii(ptr addrspace(1) %load_pointer, i1 %predicate, i32 %default_value, i32 0)
  call spir_func void @_Z28__spirv_PredicatedStoreINTELPU3AS1Kiib(ptr addrspace(1) %store_pointer, i32 %store_object, i1 %predicate)
  call spir_func void @_Z28__spirv_PredicatedStoreINTELPU3AS1Kiibi(ptr addrspace(1) %store_pointer, i32 %store_object, i1 %predicate, i32 0)
  ret void
}

declare spir_func i32 @_Z27__spirv_PredicatedLoadINTELPU3AS1Kibi(ptr addrspace(1), i1, i32)
declare spir_func i32 @_Z27__spirv_PredicatedLoadINTELPU3AS1Kibii(ptr addrspace(1), i1, i32, i32)
declare spir_func void @_Z28__spirv_PredicatedStoreINTELPU3AS1Kiib(ptr addrspace(1), i32, i1)
declare spir_func void @_Z28__spirv_PredicatedStoreINTELPU3AS1Kiibi(ptr addrspace(1), i32, i1, i32)

; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_predicated_io %s -o - | FileCheck %s

; CHECK-ERROR: LLVM ERROR: OpPredicated[Load/Store]INTEL
; CHECK-ERROR-SAME: instructions require the following SPIR-V extension: SPV_INTEL_predicated_io

; CHECK-DAG: Capability PredicatedIOINTEL
; CHECK-DAG: Extension "SPV_INTEL_predicated_io"

; CHECK-DAG: %[[Int32Ty:[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: %[[IntPtrTy:[0-9]+]] = OpTypePointer CrossWorkgroup %[[Int32Ty]]
; CHECK-DAG: %[[BoolTy:[0-9]+]] = OpTypeBool
; CHECK-DAG: %[[VoidTy:[0-9]+]] = OpTypeVoid
; CHECK: %[[LoadPtr:[0-9]+]] = OpFunctionParameter %[[IntPtrTy]]
; CHECK: %[[StorePtr:[0-9]+]] = OpFunctionParameter %[[IntPtrTy]]
; CHECK: %[[DefaultVal:[0-9]+]] = OpFunctionParameter %[[Int32Ty]]
; CHECK: %[[StoreObj:[0-9]+]] = OpFunctionParameter %[[Int32Ty]]
; CHECK: %[[Predicate:[0-9]+]] = OpFunctionParameter %[[BoolTy]]
; CHECK: PredicatedLoadINTEL %[[Int32Ty]] %[[LoadPtr]] %[[Predicate]] %[[DefaultVal]]
; CHECK: PredicatedLoadINTEL %[[Int32Ty]] %[[LoadPtr]] %[[Predicate]] %[[DefaultVal]] None
; CHECK: PredicatedStoreINTEL %[[StorePtr]] %[[StoreObj]] %[[Predicate]]
; CHECK: PredicatedStoreINTEL %[[StorePtr]] %[[StoreObj]] %[[Predicate]] None

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

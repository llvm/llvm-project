; The goal of the test case is to ensure that the Backend doesn't crash on the stage
; of type inference. Result SPIR-V is not expected to be valid from the perspective
; of spirv-val in this case, because there is a difference of accepted return types
; between atomicrmw and OpAtomicExchange.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

; CHECK-DAG: %[[#LongTy:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#PtrLongTy:]] = OpTypePointer CrossWorkgroup %[[#LongTy]]
; CHECK-DAG: %[[#IntTy:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Scope:]] = OpConstant %[[#IntTy]] 1
; CHECK-DAG: %[[#MemSem:]] = OpConstant %[[#IntTy]] 8
; CHECK-DAG: %[[#PtrPtrLongTy:]] = OpTypePointer CrossWorkgroup %[[#PtrLongTy]]

; CHECK: OpFunction
; CHECK: %[[#Arg1:]] = OpFunctionParameter %[[#PtrPtrLongTy]]
; CHECK: %[[#Arg2:]] = OpFunctionParameter %[[#PtrLongTy]]
; CHECK: OpAtomicExchange %[[#PtrLongTy]] %[[#Arg1]] %[[#Scope]] %[[#MemSem]] %[[#Arg2]]
; CHECK: OpFunctionEnd

define dso_local spir_func void @test1(ptr addrspace(1) %arg1, ptr addrspace(1) byval(i64) %arg_ptr) {
entry:
  %r = atomicrmw xchg ptr addrspace(1) %arg1, ptr addrspace(1) %arg_ptr acq_rel
  ret void
}

; CHECK: OpFunction
; CHECK: %[[#Arg3:]] = OpFunctionParameter %[[#PtrLongTy]]
; CHECK: %[[#Arg4:]] = OpFunctionParameter %[[#LongTy]]
; CHECK: OpAtomicExchange %[[#LongTy]] %[[#Arg3]] %[[#Scope]] %[[#MemSem]] %[[#Arg4]]
; CHECK: OpFunctionEnd

define dso_local spir_func void @test2(ptr addrspace(1) %arg1, i64 %arg_ptr) {
entry:
  %r = atomicrmw xchg ptr addrspace(1) %arg1, i64 %arg_ptr acq_rel
  ret void
}

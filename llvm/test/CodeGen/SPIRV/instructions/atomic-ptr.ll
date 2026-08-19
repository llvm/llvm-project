; When the exchanged value is a pointer, 'atomicrmw xchg' is lowered to
; OpAtomicExchange on integers: the value operand is converted to an integer of
; the pointer size, the pointer operand is bitcast to a pointer to that integer
; type, and the integer result is converted back to a pointer.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,SPIRV64
; RUN: %if spirv-tools %{ llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,SPIRV32
; RUN: %if spirv-tools %{ llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#LongTy:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#PtrLongTy:]] = OpTypePointer CrossWorkgroup %[[#LongTy]]
; CHECK-DAG: %[[#IntTy:]] = OpTypeInt 32 0
; SPIRV32-DAG: %[[#PtrIntTy:]] = OpTypePointer CrossWorkgroup %[[#IntTy]]
; CHECK-DAG: %[[#Scope:]] = OpConstantNull %[[#IntTy]]
; CHECK-DAG: %[[#MemSem:]] = OpConstant %[[#IntTy]] 520
; CHECK-DAG: %[[#PtrPtrLongTy:]] = OpTypePointer CrossWorkgroup %[[#PtrLongTy]]

; CHECK: OpFunction
; CHECK: %[[#Arg1:]] = OpFunctionParameter %[[#PtrPtrLongTy]]
; CHECK: %[[#Arg2:]] = OpFunctionParameter %[[#PtrLongTy]]
; SPIRV64: %[[#CastVal:]] = OpConvertPtrToU %[[#LongTy]] %[[#Arg2]]
; SPIRV64: %[[#CastPtr:]] = OpBitcast %[[#PtrLongTy]] %[[#Arg1]]
; SPIRV64: %[[#Res:]] = OpAtomicExchange %[[#LongTy]] %[[#CastPtr]] %[[#Scope]] %[[#MemSem]] %[[#CastVal]]
; SPIRV32: %[[#CastVal:]] = OpConvertPtrToU %[[#IntTy]] %[[#Arg2]]
; SPIRV32: %[[#CastPtr:]] = OpBitcast %[[#PtrIntTy]] %[[#Arg1]]
; SPIRV32: %[[#Res:]] = OpAtomicExchange %[[#IntTy]] %[[#CastPtr]] %[[#Scope]] %[[#MemSem]] %[[#CastVal]]
; CHECK: OpConvertUToPtr %[[#PtrLongTy]] %[[#Res]]
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

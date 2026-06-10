; RUN: llc -O0 -mtriple=spirv32 %s -o - | FileCheck %s --check-prefixes=CHECK,SPIRV32
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-- %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv64 %s -o - | FileCheck %s --check-prefixes=CHECK,SPIRV64
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-- %s -o - -filetype=obj | spirv-val %}


; Check that 'load atomic' LLVM IR instructions are lowered correctly to
; OpAtomicLoad when a pointer is passed as value.

; SPIRV32-DAG: %[[#Int32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Int8:]] = OpTypeInt 8 0
; SPIRV32-DAG: %[[#Int32Ptr:]] = OpTypePointer Generic %[[#Int32]]
; CHECK-DAG: %[[#Int8Ptr:]] = OpTypePointer Generic %[[#Int8]]
; CHECK-DAG: %[[#PtrInt8Ptr:]] = OpTypePointer Generic %[[#Int8Ptr]]
; SPIRV64-DAG: %[[#Int64:]] = OpTypeInt 64 0
; SPIRV64-DAG: %[[#Int64Ptr:]] = OpTypePointer Generic %[[#Int64]]

define ptr addrspace(4)  @load_ptr(ptr addrspace(4) %ptr) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#Ptr:]] = OpFunctionParameter %[[#PtrInt8Ptr]]
; SPIRV32:       %[[#ConvertPtr:]] = OpBitcast %[[#Int32Ptr]] %[[#Ptr]]
; SPIRV32:       %[[#Val:]] = OpAtomicLoad %[[#Int32Ptr]] %[[#ConvertPtr]] %[[#]] %[[#]]
; SPIRV32:       %[[#ConvertVal:]] = OpConvertUToPtr %[[#Int32Ptr]] %[[#Val]]
; SPIRV64:       %[[#ConvertPtr:]] = OpBitcast %[[#Int64Ptr]] %[[#Ptr]]
; SPIRV64:       %[[#Val:]] = OpAtomicLoad %[[#Int64Ptr]] %[[#ConvertPtr]] %[[#]] %[[#]]
; SPIRV64:       %[[#ConvertVal:]] = OpConvertUToPtr %[[#Int64Ptr]] %[[#Val]]
; CHECK:       OpReturnValue %[[#ConvertVal]]
  %val = load atomic ptr addrspace(4), ptr addrspace(4) %ptr monotonic, align 8
  ret ptr addrspace(4) %val
}

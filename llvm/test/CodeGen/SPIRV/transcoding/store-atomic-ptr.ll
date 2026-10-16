; RUN: llc -O0 -mtriple=spirv32 %s -o - | FileCheck %s --check-prefixes=CHECK,SPIRV32
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-- %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv64 %s -o - | FileCheck %s --check-prefixes=CHECK,SPIRV64
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-- %s -o - -filetype=obj | spirv-val %}


; Check that 'store atomic' LLVM IR instructions are lowered correctly to
; OpAtomicStore when a pointer is passed as value.

; SPIRV32-DAG: %[[#Int32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Int8:]] = OpTypeInt 8 0
; SPIRV32-DAG: %[[#Int32Ptr:]] = OpTypePointer Generic %[[#Int32]]
; CHECK-DAG: %[[#Int8Ptr:]] = OpTypePointer Generic %[[#Int8]]
; CHECK-DAG: %[[#PtrInt8Ptr:]] = OpTypePointer Generic %[[#Int8Ptr]]
; SPIRV64-DAG: %[[#Int64:]] = OpTypeInt 64 0
; SPIRV64-DAG: %[[#Int64Ptr:]] = OpTypePointer Generic %[[#Int64]]

define void @store_ptr(ptr addrspace(4) %ptr, ptr addrspace(4) %val) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#Ptr:]] = OpFunctionParameter %[[#PtrInt8Ptr]]
; CHECK:       %[[#Val:]] = OpFunctionParameter %[[#Int8Ptr]]
; SPIRV32:       %[[#ConvertVal:]] = OpConvertPtrToU %[[#Int32]] %[[#Val]]
; SPIRV32:       %[[#ConvertPtr:]] = OpBitcast %[[#Int32Ptr]] %[[#Ptr]]
; SPIRV64:       %[[#ConvertVal:]] = OpConvertPtrToU %[[#Int64]] %[[#Val]]
; SPIRV64:       %[[#ConvertPtr:]] = OpBitcast %[[#Int64Ptr]] %[[#Ptr]]
; CHECK:       OpAtomicStore %[[#ConvertPtr]] %[[#]] %[[#]] %[[#ConvertVal]]
; CHECK:       OpReturn
  store atomic ptr addrspace(4) %val, ptr addrspace(4) %ptr monotonic, align 8
  ret void
}

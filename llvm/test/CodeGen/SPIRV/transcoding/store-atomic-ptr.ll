; RUN: llc -O0 -mtriple=spirv32-- %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-- %s -o - -filetype=obj | spirv-val %}

; Check that 'store atomic' LLVM IR instructions are lowered correctly to
; OpAtomicStore when a pointer is passed as value

; CHECK-DAG: %[[#Int32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Int8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#Int32Ptr:]] = OpTypePointer Generic %[[#Int32]]
; CHECK-DAG: %[[#Int8Ptr:]] = OpTypePointer Generic %[[#Int8]]
; CHECK-DAG: %[[#PtrInt8Ptr:]] = OpTypePointer Generic %[[#Int8Ptr]]

define void @store_ptr(ptr addrspace(4) %ptr, ptr addrspace(4) %val) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#ptr:]] = OpFunctionParameter %[[#PtrInt8Ptr]]
; CHECK:       %[[#val:]] = OpFunctionParameter %[[#Int8Ptr]]
; CHECK:       %[[#convertVal:]] = OpConvertPtrToU %[[#Int32:]] %[[#val:]]
; CHECK:       %[[#convertPtr:]] = OpBitcast %[[#Int32Ptr:]] %[[#ptr:]]
; CHECK:       OpAtomicStore %[[#convertPtr]] %[[#]] %[[#]] %[[#convertVal]]
; CHECK:       OpReturn
  store atomic ptr addrspace(4) %val, ptr addrspace(4) %ptr monotonic, align 8
  ret void
}
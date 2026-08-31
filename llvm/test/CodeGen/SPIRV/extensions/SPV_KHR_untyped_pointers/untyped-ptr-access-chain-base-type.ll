; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}

; When an untyped pointer is used first with one element type and then indexed
; as a struct through a multi-index GEP, the struct access chain must use the
; struct as its Base Type, not the pointer's first-deduced i8 type. An i8 Base
; Type with member indices is invalid SPIR-V and reverse-translates to an
; invalid multi-index getelementptr i8.

%struct.inner = type { i32, i32 }
%struct.middle = type { i64, i32, %struct.inner }
%struct.outer = type { i32, i32, i32, %struct.middle }

; CHECK-DAG: %[[#I8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#I64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#INNER:]] = OpTypeStruct %[[#I32]] %[[#I32]]
; CHECK-DAG: %[[#MIDDLE:]] = OpTypeStruct %[[#I64]] %[[#I32]] %[[#INNER]]
; CHECK-DAG: %[[#OUTER:]] = OpTypeStruct %[[#I32]] %[[#I32]] %[[#I32]] %[[#MIDDLE]]

; The i8 access keeps an i8 Base Type with a single index.
; CHECK: OpUntypedInBoundsPtrAccessChainKHR %[[#]] %[[#I8]] %[[#PTR:]] %[[#]]
; The struct accesses on the same pointer must use the struct Base Types, not i8.
; CHECK: OpUntypedInBoundsPtrAccessChainKHR %[[#]] %[[#OUTER]] %[[#PTR]] %[[#]] %[[#]]
; CHECK: OpUntypedInBoundsPtrAccessChainKHR %[[#]] %[[#MIDDLE]] %[[#]] %[[#]] %[[#]]
; CHECK: OpUntypedInBoundsPtrAccessChainKHR %[[#]] %[[#INNER]] %[[#]] %[[#]] %[[#]]
define spir_kernel void @i8_use_then_struct_chain(ptr addrspace(4) %p, ptr addrspace(1) %out) #0 {
entry:
  %b = getelementptr inbounds i8, ptr addrspace(4) %p, i64 0
  store i8 7, ptr addrspace(4) %b, align 1
  %gep1 = getelementptr inbounds %struct.outer, ptr addrspace(4) %p, i32 0, i32 3
  %gep2 = getelementptr inbounds %struct.middle, ptr addrspace(4) %gep1, i32 0, i32 2
  %gep3 = getelementptr inbounds %struct.inner, ptr addrspace(4) %gep2, i32 0, i32 1
  %val = load i32, ptr addrspace(4) %gep3, align 4
  store i32 %val, ptr addrspace(1) %out, align 4
  ret void
}

attributes #0 = { nounwind }

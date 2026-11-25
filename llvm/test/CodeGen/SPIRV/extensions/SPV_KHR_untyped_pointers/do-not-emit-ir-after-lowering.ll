; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[#GenericUntypedPtrTy:]] = OpTypeUntypedPointerKHR Generic
; CHECK-NEXT: %[[#UnionTy:]] = OpTypeStruct %[[#GenericUntypedPtrTy]]
; CHECK-NEXT: %[[#StructTy:]] = OpTypeStruct %[[#UnionTy]]
; CHECK-NEXT: %[[#ConstantStructPtrTy:]] = OpTypePointer UniformConstant %[[#StructTy]]
; CHECK: %[[#ConstantUntypedStructPtrTy:]] = OpTypeUntypedPointerKHR UniformConstant
; CHECK: %[[#PtrToStruct:]] = OpFunctionParameter %[[#ConstantStructPtrTy]]
; CHECK: %[[#BC:]] = OpBitcast %[[#ConstantUntypedStructPtrTy]] %[[#PtrToStruct]]
; CHECK: %[[#]] = OpLoad %[[#GenericUntypedPtrTy]] %[[#BC]]

%struct.s = type { %union.u }
%union.u = type { ptr addrspace(4) }

define spir_kernel void @k(ptr addrspace(2) byval(%struct.s) %0) {
entry:
  %1 = load ptr addrspace(4), ptr addrspace(2) %0, align 8
  br i1 false, label %cond.true, label %cond.false

cond.true:
  store ptr addrspace(4) %1, ptr addrspace(4) null, align 4
  ret void

cond.false:
  %call5 = call spir_func [2 x i32] @f(ptr addrspace(4) %1)
  ret void
}

define spir_func [2 x i32] @f(ptr addrspace(4) %x) {
entry:
  %0 = getelementptr [8 x i8], ptr addrspace(4) %x, i64 0
  ret [2 x i32] zeroinitializer
}

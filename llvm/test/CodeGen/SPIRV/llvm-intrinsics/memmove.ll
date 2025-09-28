; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-NOT: llvm.memmove

; CHECK-DAG: %[[#Int8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#Int32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Int64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#Ptr_CrossWG_8:]] = OpTypePointer CrossWorkgroup %[[#Int8]]
; CHECK-DAG: %[[#Ptr_Generic_32:]] = OpTypePointer Generic %[[#Int32]]
; CHECK-DAG: %[[#Const_64:]] = OpConstant %[[#Int32]] 64
; CHECK-DAG: %[[#Const_36:]] = OpConstant %[[#Int32]] 36
; CHECK-DAG: %[[#Const_30:]] = OpConstant %[[#Int32]] 30
; CHECK-DAG: %[[#Const_32_64:]] = OpConstant %[[#Int64]] 32

; CHECK: %[[#Param1:]] = OpFunctionParameter %[[#Ptr_CrossWG_8]]
; CHECK: %[[#Param2:]] = OpFunctionParameter %[[#Ptr_CrossWG_8]]
; CHECK: %[[#Size1:]] = OpUConvert %[[#Int64]] %[[#Const_64]]
; CHECK: OpCopyMemorySized %[[#Param2]] %[[#Param1]] %[[#Size1]] Aligned 64

; CHECK: %[[#Src:]] = OpFunctionParameter %[[#Ptr_CrossWG_8]]
; CHECK: %[[#CastDst2:]] = OpGenericCastToPtr %[[#Ptr_CrossWG_8]] %[[#GenPtr:]]
; CHECK: %[[#Size2:]] = OpUConvert %[[#Int64]] %[[#Const_36]]
; CHECK: OpCopyMemorySized %[[#CastDst2]] %[[#Src]] %[[#Size2]] Aligned 64

; CHECK: %[[#Param1:]] = OpFunctionParameter %[[#Ptr_CrossWG_8]]
; CHECK: %[[#Param2:]] = OpFunctionParameter %[[#Ptr_CrossWG_8]]
; CHECK: %[[#Size3:]] = OpUConvert %[[#Int64]] %[[#Const_30]]
; CHECK: OpCopyMemorySized %[[#Param2]] %[[#Param1]] %[[#Size3]] Aligned 1

; CHECK: %[[#Phi:]] = OpPhi %[[#Ptr_Generic_32]] %[[#Op1:]] %[[#Lbl1:]] %[[#Op2:]] %[[#Lbl2:]]
; CHECK: %[[#Cast:]] = OpPtrCastToGeneric %[[#]] %[[#]]
; CHECK: OpCopyMemorySized %[[#Cast]] %[[#Phi]] %[[#Const_32_64]] Aligned 8

%struct.SomeStruct = type { <16 x float>, i32, [60 x i8] }
%class.kfunc = type <{ i32, i32, i32, [4 x i8] }>

@InvocIndex = external local_unnamed_addr addrspace(1) constant i64, align 8
@"func_object1" = internal addrspace(3) global %class.kfunc zeroinitializer, align 8

define spir_kernel void @test_full_move(%struct.SomeStruct addrspace(1)* captures(none) readonly %in, %struct.SomeStruct addrspace(1)* captures(none) %out) {
  %1 = bitcast %struct.SomeStruct addrspace(1)* %in to i8 addrspace(1)*
  %2 = bitcast %struct.SomeStruct addrspace(1)* %out to i8 addrspace(1)*
  call void @llvm.memmove.p1i8.p1i8.i32(i8 addrspace(1)* align 64 %2, i8 addrspace(1)* align 64 %1, i32 64, i1 false)
  ret void
}

define spir_kernel void @test_partial_move(%struct.SomeStruct addrspace(1)* captures(none) readonly %in, %struct.SomeStruct addrspace(4)* captures(none) %out) {
  %1 = bitcast %struct.SomeStruct addrspace(1)* %in to i8 addrspace(1)*
  %2 = bitcast %struct.SomeStruct addrspace(4)* %out to i8 addrspace(4)*
  %3 = addrspacecast i8 addrspace(4)* %2 to i8 addrspace(1)*
  call void @llvm.memmove.p1i8.p1i8.i32(i8 addrspace(1)* align 64 %3, i8 addrspace(1)* align 64 %1, i32 36, i1 false)
  ret void
}

define spir_kernel void @test_array(i8 addrspace(1)* %in, i8 addrspace(1)* %out) {
  call void @llvm.memmove.p1i8.p1i8.i32(i8 addrspace(1)* %out, i8 addrspace(1)* %in, i32 30, i1 false)
  ret void
}

define weak_odr dso_local spir_kernel void @test_phi() local_unnamed_addr {
entry:
  %0 = alloca i32, align 8
  %1 = addrspacecast i32* %0 to i32 addrspace(4)*
  %2 = load i64, i64 addrspace(1)* @InvocIndex, align 8
  %cmp = icmp eq i64 %2, 0
  br i1 %cmp, label %leader, label %entry.merge_crit_edge

entry.merge_crit_edge:                            ; preds = %entry
  %3 = bitcast i32 addrspace(4)* %1 to i8 addrspace(4)*
  br label %merge

leader:                                           ; preds = %entry
  %4 = bitcast i32 addrspace(4)* %1 to i8 addrspace(4)*
  br label %merge

merge:                                            ; preds = %entry.merge_crit_edge, %leader
  %phi = phi i8 addrspace(4)* [ %3, %entry.merge_crit_edge ], [ %4, %leader ]
  %5 = addrspacecast i8 addrspace(3)* bitcast (%class.kfunc addrspace(3)* @"func_object1" to i8 addrspace(3)*) to i8 addrspace(4)*
  call void @llvm.memmove.p4i8.p4i8.i64(i8 addrspace(4)* align 8 dereferenceable(32) %5, i8 addrspace(4)* align 8 dereferenceable(32) %phi, i64 32, i1 false)
  ret void
}

declare void @llvm.memmove.p4i8.p4i8.i64(i8 addrspace(4)* captures(none) writeonly, i8 addrspace(4)* captures(none) readonly, i64, i1 immarg)

declare void @llvm.memmove.p1i8.p1i8.i32(i8 addrspace(1)* captures(none), i8 addrspace(1)* captures(none) readonly, i32, i1)

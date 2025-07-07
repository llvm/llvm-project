; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o - | FileCheck %s
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName %[[I9:.*]] "_ZN13BaseIncrement9incrementEPi"
; CHECK-DAG: OpName %[[I29:.*]] "_ZN12IncrementBy29incrementEPi"
; CHECK-DAG: OpName %[[I49:.*]] "_ZN12IncrementBy49incrementEPi"
; CHECK-DAG: OpName %[[I89:.*]] "_ZN12IncrementBy89incrementEPi"
; CHECK-DAG: OpName %[[Foo:.*]] "foo"

; CHECK-DAG: %[[TyVoid:.*]] = OpTypeVoid
; CHECK-DAG: %[[TyInt32:.*]] = OpTypeInt 32 0
; CHECK-DAG: %[[TyInt8:.*]] = OpTypeInt 8 0
; CHECK-DAG: %[[Const8:.*]] = OpConstant %[[TyInt32]] 8
; CHECK-DAG: %[[TyArr:.*]] = OpTypeArray %[[TyInt8]] %[[Const8]]
; CHECK-DAG: %[[TyStruct1:.*]] = OpTypeStruct %[[TyArr]]
; CHECK-DAG: %[[TyStruct2:.*]] = OpTypeStruct %[[TyStruct1]]
; CHECK-DAG: %[[TyPtrStruct2:.*]] = OpTypePointer Generic %[[TyStruct2]]
; CHECK-DAG: %[[TyFun:.*]] = OpTypeFunction %[[TyVoid]] %[[TyPtrStruct2]] %[[#]]
; CHECK-DAG: %[[TyPtrFun:.*]] = OpTypePointer Generic %[[TyFun]]
; CHECK-DAG: %[[TyPtrPtrFun:.*]] = OpTypePointer Generic %[[TyPtrFun]]

; CHECK-DAG: %[[I9]] = OpFunction
; CHECK-DAG: %[[I29]] = OpFunction
; CHECK-DAG: %[[I49]] = OpFunction
; CHECK-DAG: %[[I89]] = OpFunction

; CHECK: %[[Foo]] = OpFunction
; CHECK-4: OpFunctionParameter

; CHECK: %[[Arg1:.*]] = OpPhi %[[TyPtrStruct2]]
; CHECK: %[[VTbl:.*]] = OpBitcast %[[TyPtrPtrFun]] %[[#]]
; CHECK: %[[FP:.*]] = OpLoad %[[TyPtrFun]] %[[VTbl]]
; CHECK: %[[#]] = OpFunctionPointerCallINTEL %[[TyVoid]] %[[FP]] %[[Arg1]] %[[#]]

; CHECK-NO: OpFunction

%"cls::id" = type { %"cls::detail::array" }
%"cls::detail::array" = type { [1 x i64] }
%struct.obj_storage_t = type { %"struct.aligned_storage<BaseIncrement, IncrementBy2, IncrementBy4, IncrementBy8>::type" }
%"struct.aligned_storage<BaseIncrement, IncrementBy2, IncrementBy4, IncrementBy8>::type" = type { [8 x i8] }

@_ZTV12IncrementBy8 = linkonce_odr dso_local unnamed_addr addrspace(1) constant { [3 x ptr addrspace(4)] } { [3 x ptr addrspace(4)] [ptr addrspace(4) null, ptr addrspace(4) null, ptr addrspace(4) addrspacecast (ptr @_ZN12IncrementBy89incrementEPi to ptr addrspace(4))] }, align 8
@_ZTV13BaseIncrement = linkonce_odr dso_local unnamed_addr addrspace(1) constant { [3 x ptr addrspace(4)] } { [3 x ptr addrspace(4)] [ptr addrspace(4) null, ptr addrspace(4) null, ptr addrspace(4) addrspacecast (ptr @_ZN13BaseIncrement9incrementEPi to ptr addrspace(4))] }, align 8
@_ZTV12IncrementBy4 = linkonce_odr dso_local unnamed_addr addrspace(1) constant { [3 x ptr addrspace(4)] } { [3 x ptr addrspace(4)] [ptr addrspace(4) null, ptr addrspace(4) null, ptr addrspace(4) addrspacecast (ptr @_ZN12IncrementBy49incrementEPi to ptr addrspace(4))] }, align 8
@_ZTV12IncrementBy2 = linkonce_odr dso_local unnamed_addr addrspace(1) constant { [3 x ptr addrspace(4)] } { [3 x ptr addrspace(4)] [ptr addrspace(4) null, ptr addrspace(4) null, ptr addrspace(4) addrspacecast (ptr @_ZN12IncrementBy29incrementEPi to ptr addrspace(4))] }, align 8
@__spirv_BuiltInWorkgroupId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInGlobalLinearId = external dso_local local_unnamed_addr addrspace(1) constant i64, align 8
@__spirv_BuiltInWorkgroupSize = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32

define weak_odr dso_local spir_kernel void @foo(ptr addrspace(1) noundef align 8 %_arg_StorageAcc, ptr noundef byval(%"cls::id") align 8 %_arg_StorageAcc3, i32 noundef %_arg_TestCase, ptr addrspace(1) noundef align 4 %_arg_DataAcc) {
entry:
  %r0 = load i64, ptr %_arg_StorageAcc3, align 8
  %add.ptr.i = getelementptr inbounds %struct.obj_storage_t, ptr addrspace(1) %_arg_StorageAcc, i64 %r0
  %arrayidx.ascast.i = addrspacecast ptr addrspace(1) %add.ptr.i to ptr addrspace(4)
  %cmp.i = icmp ugt i32 %_arg_TestCase, 3
  br i1 %cmp.i, label %entry.critedge, label %if.end.1

entry.critedge: ; preds = %entry
  %vtable.i.pre = load ptr addrspace(4), ptr addrspace(4) null, align 8
  br label %exit

if.end.1:                                         ; preds = %entry
  switch i32 %_arg_TestCase, label %if.end.5 [
    i32 0, label %if.end.2
    i32 1, label %if.end.3
    i32 2, label %if.end.4
  ]

if.end.5:                                 ; preds = %if.end.1
  store ptr addrspace(1) getelementptr inbounds inrange(-16, 8) (i8, ptr addrspace(1) @_ZTV12IncrementBy8, i64 16), ptr addrspace(1) %add.ptr.i, align 8
  br label %exit

if.end.4:                                   ; preds = %if.end.1
  store ptr addrspace(1) getelementptr inbounds inrange(-16, 8) (i8, ptr addrspace(1) @_ZTV12IncrementBy4, i64 16), ptr addrspace(1) %add.ptr.i, align 8
  br label %exit

if.end.3:                                     ; preds = %if.end.1
  store ptr addrspace(1) getelementptr inbounds inrange(-16, 8) (i8, ptr addrspace(1) @_ZTV12IncrementBy2, i64 16), ptr addrspace(1) %add.ptr.i, align 8
  br label %exit

if.end.2:                                       ; preds = %if.end.1
  store ptr addrspace(1) getelementptr inbounds inrange(-16, 8) (i8, ptr addrspace(1) @_ZTV13BaseIncrement, i64 16), ptr addrspace(1) %add.ptr.i, align 8
  br label %exit

exit: ; preds = %if.end.2, %if.end.3, %if.end.4, %if.end.5, %entry.critedge
  %vtable.i = phi ptr addrspace(4) [ %vtable.i.pre, %entry.critedge ], [ inttoptr (i64 ptrtoint (ptr addrspace(1) getelementptr inbounds inrange(-16, 8) (i8, ptr addrspace(1) @_ZTV12IncrementBy8, i64 16) to i64) to ptr addrspace(4)), %if.end.5 ], [ inttoptr (i64 ptrtoint (ptr addrspace(1) getelementptr inbounds inrange(-16, 8) (i8, ptr addrspace(1) @_ZTV12IncrementBy4, i64 16) to i64) to ptr addrspace(4)), %if.end.4 ], [ inttoptr (i64 ptrtoint (ptr addrspace(1) getelementptr inbounds inrange(-16, 8) (i8, ptr addrspace(1) @_ZTV12IncrementBy2, i64 16) to i64) to ptr addrspace(4)), %if.end.3 ], [ inttoptr (i64 ptrtoint (ptr addrspace(1) getelementptr inbounds inrange(-16, 8) (i8, ptr addrspace(1) @_ZTV13BaseIncrement, i64 16) to i64) to ptr addrspace(4)), %if.end.2 ]
  %retval.0.i = phi ptr addrspace(4) [ null, %entry.critedge ], [ %arrayidx.ascast.i, %if.end.5 ], [ %arrayidx.ascast.i, %if.end.4 ], [ %arrayidx.ascast.i, %if.end.3 ], [ %arrayidx.ascast.i, %if.end.2 ]
  %r1 = addrspacecast ptr addrspace(1) %_arg_DataAcc to ptr addrspace(4)
  %r2 = load ptr addrspace(4), ptr addrspace(4) %vtable.i, align 8
  tail call spir_func addrspace(4) void %r2(ptr addrspace(4) noundef align 8 dereferenceable_or_null(8) %retval.0.i, ptr addrspace(4) noundef %r1)
  ret void
}

declare dso_local spir_func void @_ZN13BaseIncrement9incrementEPi(ptr addrspace(4) noundef align 8 dereferenceable_or_null(8), ptr addrspace(4) noundef)
declare dso_local spir_func void @_ZN12IncrementBy29incrementEPi(ptr addrspace(4) noundef align 8 dereferenceable_or_null(8), ptr addrspace(4) noundef)
declare dso_local spir_func void @_ZN12IncrementBy49incrementEPi(ptr addrspace(4) noundef align 8 dereferenceable_or_null(8), ptr addrspace(4) noundef)
declare dso_local spir_func void @_ZN12IncrementBy89incrementEPi(ptr addrspace(4) noundef align 8 dereferenceable_or_null(8), ptr addrspace(4) noundef)

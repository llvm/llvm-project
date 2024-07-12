; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-DAG: %[[#Char:]] = OpTypeInt 8 0
; CHECK-SPIRV-DAG: %[[#GlobalCharPtr:]] = OpTypePointer CrossWorkgroup %[[#Char]]
; CHECK-SPIRV-DAG: %[[#LocalCharPtr:]] = OpTypePointer Workgroup %[[#Char]]
; CHECK-SPIRV-DAG: %[[#PrivateCharPtr:]] = OpTypePointer Function %[[#Char]]
; CHECK-SPIRV-DAG: %[[#GenericCharPtr:]] = OpTypePointer Generic %[[#Char]]

; CHECK-SPIRV-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: %[[#GlobalIntPtr:]] = OpTypePointer CrossWorkgroup %[[#Int]]
; CHECK-SPIRV-DAG: %[[#PrivateIntPtr:]] = OpTypePointer Function %[[#Int]]
; CHECK-SPIRV-DAG: %[[#GenericIntPtr:]] = OpTypePointer Generic %[[#Int]]

%id = type { %arr }
%arr = type { [1 x i64] }

@__spirv_BuiltInGlobalInvocationId = external local_unnamed_addr addrspace(1) constant <3 x i64>

; Mangling

; CHECK-SPIRV:      OpFunction
; CHECK-SPIRV:      OpPtrCastToGeneric %[[#GenericIntPtr]]
; CHECK-SPIRV-NEXT: OpPtrCastToGeneric %[[#GenericCharPtr]]
; CHECK-SPIRV-NEXT: OpPtrCastToGeneric %[[#GenericIntPtr]]
; CHECK-SPIRV-NEXT: OpGenericCastToPtr %[[#GlobalIntPtr]]
; CHECK-SPIRV-NEXT: OpGenericCastToPtr %[[#LocalCharPtr]]
; CHECK-SPIRV-NEXT: OpGenericCastToPtr %[[#PrivateIntPtr]]
; CHECK-SPIRV-NEXT: OpGenericCastToPtr %[[#GlobalIntPtr]]
; CHECK-SPIRV-NEXT: OpGenericCastToPtr %[[#LocalCharPtr]]
; CHECK-SPIRV-NEXT: OpGenericCastToPtr %[[#PrivateIntPtr]]
; CHECK-SPIRV:      OpFunctionEnd

define spir_kernel void @test1(ptr addrspace(1) %_arg_GlobalA, ptr byval(%id) %_arg_GlobalId, ptr addrspace(3) %_arg_LocalA) {
entry:
  %var = alloca i32
  %p0 = load i64, ptr %_arg_GlobalId
  %add = getelementptr inbounds i32, ptr addrspace(1) %_arg_GlobalA, i64 %p0
  %p2 = load i64, ptr addrspace(1) @__spirv_BuiltInGlobalInvocationId
  %idx = getelementptr inbounds i32, ptr addrspace(1) %add, i64 %p2
  %var1 = addrspacecast ptr addrspace(1) %idx to ptr addrspace(4)
  %var2 = addrspacecast ptr addrspace(3) %_arg_LocalA to ptr addrspace(4)
  %var3 = addrspacecast ptr %var to ptr addrspace(4)
  %G = call spir_func ptr addrspace(1) @_Z33__spirv_GenericCastToPtr_ToGlobalPvi(ptr addrspace(4) %var1, i32 5)
  %L = call spir_func ptr addrspace(3) @_Z32__spirv_GenericCastToPtr_ToLocalPvi(ptr addrspace(4) %var2, i32 4)
  %P = call spir_func ptr @_Z34__spirv_GenericCastToPtr_ToPrivatePvi(ptr addrspace(4) %var3, i32 7)
  %GE = call spir_func ptr addrspace(1) @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi(ptr addrspace(4) %var1, i32 5)
  %LE = call spir_func ptr addrspace(3) @_Z40__spirv_GenericCastToPtrExplicit_ToLocalPvi(ptr addrspace(4) %var2, i32 4)
  %PE = call spir_func ptr @_Z42__spirv_GenericCastToPtrExplicit_ToPrivatePvi(ptr addrspace(4) %var3, i32 7)
  ret void
}

; CHECK-SPIRV:      OpFunction
; CHECK-SPIRV:      OpPtrCastToGeneric %[[#GenericIntPtr]]
; CHECK-SPIRV-NEXT: OpPtrCastToGeneric %[[#GenericCharPtr]]
; CHECK-SPIRV-NEXT: OpPtrCastToGeneric %[[#GenericIntPtr]]
; CHECK-SPIRV-NEXT: OpGenericCastToPtr %[[#GlobalIntPtr]]
; CHECK-SPIRV-NEXT: OpGenericCastToPtr %[[#LocalCharPtr]]
; CHECK-SPIRV-NEXT: OpGenericCastToPtr %[[#PrivateIntPtr]]
; CHECK-SPIRV:      OpFunctionEnd

define spir_kernel void @test2(ptr addrspace(1) %_arg_GlobalA, ptr byval(%id) %_arg_GlobalId, ptr addrspace(3) %_arg_LocalA) {
entry:
  %var = alloca i32
  %p0 = load i64, ptr %_arg_GlobalId
  %add = getelementptr inbounds i32, ptr addrspace(1) %_arg_GlobalA, i64 %p0
  %p2 = load i64, ptr addrspace(1) @__spirv_BuiltInGlobalInvocationId
  %idx = getelementptr inbounds i32, ptr addrspace(1) %add, i64 %p2
  %var1 = addrspacecast ptr addrspace(1) %idx to ptr addrspace(4)
  %var2 = addrspacecast ptr addrspace(3) %_arg_LocalA to ptr addrspace(4)
  %var3 = addrspacecast ptr %var to ptr addrspace(4)
  %G = call spir_func ptr addrspace(1) @_Z9to_globalPv(ptr addrspace(4) %var1)
  %L = call spir_func ptr addrspace(3) @_Z8to_localPv(ptr addrspace(4) %var2)
  %P = call spir_func ptr @_Z10to_privatePv(ptr addrspace(4) %var3)
  ret void
}

declare spir_func ptr addrspace(1) @_Z33__spirv_GenericCastToPtr_ToGlobalPvi(ptr addrspace(4), i32)
declare spir_func ptr addrspace(3) @_Z32__spirv_GenericCastToPtr_ToLocalPvi(ptr addrspace(4), i32)
declare spir_func ptr @_Z34__spirv_GenericCastToPtr_ToPrivatePvi(ptr addrspace(4), i32)
declare spir_func ptr addrspace(1) @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi(ptr addrspace(4), i32)
declare spir_func ptr addrspace(3) @_Z40__spirv_GenericCastToPtrExplicit_ToLocalPvi(ptr addrspace(4), i32)
declare spir_func ptr @_Z42__spirv_GenericCastToPtrExplicit_ToPrivatePvi(ptr addrspace(4), i32)

declare spir_func ptr addrspace(1) @_Z9to_globalPv(ptr addrspace(4))
declare spir_func ptr addrspace(3) @_Z8to_localPv(ptr addrspace(4))
declare spir_func ptr @_Z10to_privatePv(ptr addrspace(4))

; No mangling

; CHECK-SPIRV:      OpFunction
; CHECK-SPIRV:      OpPtrCastToGeneric %[[#GenericIntPtr]]
; CHECK-SPIRV-NEXT: OpPtrCastToGeneric %[[#GenericCharPtr]]
; CHECK-SPIRV-NEXT: OpPtrCastToGeneric %[[#GenericIntPtr]]
; CHECK-SPIRV-NEXT: OpGenericCastToPtr %[[#GlobalIntPtr]]
; CHECK-SPIRV-NEXT: OpGenericCastToPtr %[[#LocalCharPtr]]
; CHECK-SPIRV-NEXT: OpGenericCastToPtr %[[#PrivateIntPtr]]
; CHECK-SPIRV-NEXT: OpGenericCastToPtr %[[#GlobalIntPtr]]
; CHECK-SPIRV-NEXT: OpGenericCastToPtr %[[#LocalCharPtr]]
; CHECK-SPIRV-NEXT: OpGenericCastToPtr %[[#PrivateIntPtr]]
; CHECK-SPIRV:      OpFunctionEnd

; CHECK-SPIRV:      OpFunction
; CHECK-SPIRV:      OpPtrCastToGeneric %[[#GenericIntPtr]]
; CHECK-SPIRV-NEXT: OpPtrCastToGeneric %[[#GenericCharPtr]]
; CHECK-SPIRV-NEXT: OpPtrCastToGeneric %[[#GenericIntPtr]]
; CHECK-SPIRV-NEXT: OpGenericCastToPtr %[[#GlobalIntPtr]]
; CHECK-SPIRV-NEXT: OpGenericCastToPtr %[[#LocalCharPtr]]
; CHECK-SPIRV-NEXT: OpGenericCastToPtr %[[#PrivateIntPtr]]
; CHECK-SPIRV:      OpFunctionEnd

define spir_kernel void @test3(ptr addrspace(1) %_arg_GlobalA, ptr byval(%id) %_arg_GlobalId, ptr addrspace(3) %_arg_LocalA) {
entry:
  %var = alloca i32
  %p0 = load i64, ptr %_arg_GlobalId
  %add = getelementptr inbounds i32, ptr addrspace(1) %_arg_GlobalA, i64 %p0
  %p2 = load i64, ptr addrspace(1) @__spirv_BuiltInGlobalInvocationId
  %idx = getelementptr inbounds i32, ptr addrspace(1) %add, i64 %p2
  %var1 = addrspacecast ptr addrspace(1) %idx to ptr addrspace(4)
  %var2 = addrspacecast ptr addrspace(3) %_arg_LocalA to ptr addrspace(4)
  %var3 = addrspacecast ptr %var to ptr addrspace(4)
  %G = call spir_func ptr addrspace(1) @__spirv_GenericCastToPtr_ToGlobal(ptr addrspace(4) %var1, i32 5)
  %L = call spir_func ptr addrspace(3) @__spirv_GenericCastToPtr_ToLocal(ptr addrspace(4) %var2, i32 4)
  %P = call spir_func ptr @__spirv_GenericCastToPtr_ToPrivate(ptr addrspace(4) %var3, i32 7)
  %GE = call spir_func ptr addrspace(1) @__spirv_GenericCastToPtrExplicit_ToGlobal(ptr addrspace(4) %var1, i32 5)
  %LE = call spir_func ptr addrspace(3) @__spirv_GenericCastToPtrExplicit_ToLocal(ptr addrspace(4) %var2, i32 4)
  %PE = call spir_func ptr @__spirv_GenericCastToPtrExplicit_ToPrivate(ptr addrspace(4) %var3, i32 7)
  ret void
}

define spir_kernel void @test4(ptr addrspace(1) %_arg_GlobalA, ptr byval(%id) %_arg_GlobalId, ptr addrspace(3) %_arg_LocalA) {
entry:
  %var = alloca i32
  %p0 = load i64, ptr %_arg_GlobalId
  %add = getelementptr inbounds i32, ptr addrspace(1) %_arg_GlobalA, i64 %p0
  %p2 = load i64, ptr addrspace(1) @__spirv_BuiltInGlobalInvocationId
  %idx = getelementptr inbounds i32, ptr addrspace(1) %add, i64 %p2
  %var1 = addrspacecast ptr addrspace(1) %idx to ptr addrspace(4)
  %var2 = addrspacecast ptr addrspace(3) %_arg_LocalA to ptr addrspace(4)
  %var3 = addrspacecast ptr %var to ptr addrspace(4)
  %G = call spir_func ptr addrspace(1) @to_global(ptr addrspace(4) %var1)
  %L = call spir_func ptr addrspace(3) @to_local(ptr addrspace(4) %var2)
  %P = call spir_func ptr @to_private(ptr addrspace(4) %var3)
  ret void
}

declare spir_func ptr addrspace(1) @__spirv_GenericCastToPtr_ToGlobal(ptr addrspace(4), i32)
declare spir_func ptr addrspace(3) @__spirv_GenericCastToPtr_ToLocal(ptr addrspace(4), i32)
declare spir_func ptr @__spirv_GenericCastToPtr_ToPrivate(ptr addrspace(4), i32)
declare spir_func ptr addrspace(1) @__spirv_GenericCastToPtrExplicit_ToGlobal(ptr addrspace(4), i32)
declare spir_func ptr addrspace(3) @__spirv_GenericCastToPtrExplicit_ToLocal(ptr addrspace(4), i32)
declare spir_func ptr @__spirv_GenericCastToPtrExplicit_ToPrivate(ptr addrspace(4), i32)

declare spir_func ptr addrspace(1) @to_global(ptr addrspace(4))
declare spir_func ptr addrspace(3) @to_local(ptr addrspace(4))
declare spir_func ptr @to_private(ptr addrspace(4))

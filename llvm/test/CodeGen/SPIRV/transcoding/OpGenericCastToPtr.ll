; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-DAG: %[[#Char:]] = OpTypeInt 8 0
; CHECK-SPIRV-DAG: %[[#GlobalPtr:]] = OpTypePointer CrossWorkgroup %[[#Char]]
; CHECK-SPIRV-DAG: %[[#LocalPtr:]] = OpTypePointer Workgroup %[[#Char]]
; CHECK-SPIRV-DAG: %[[#PrivatePtr:]] = OpTypePointer Function %[[#Char]]

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: OpGenericCastToPtr %[[#GlobalPtr]]
; CHECK-SPIRV: OpGenericCastToPtr %[[#LocalPtr]]
; CHECK-SPIRV: OpGenericCastToPtr %[[#PrivatePtr]]
; CHECK-SPIRV: OpFunctionEnd

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: OpGenericCastToPtr %[[#GlobalPtr]]
; CHECK-SPIRV: OpGenericCastToPtr %[[#LocalPtr]]
; CHECK-SPIRV: OpGenericCastToPtr %[[#PrivatePtr]]
; CHECK-SPIRV: OpFunctionEnd

%id = type { %arr }
%arr = type { [1 x i64] }

@__spirv_BuiltInGlobalInvocationId = external local_unnamed_addr addrspace(1) constant <3 x i64>

define spir_kernel void @foo(ptr addrspace(1) %_arg_GlobalA, ptr byval(%id) %_arg_GlobalId, ptr addrspace(3) %_arg_LocalA) {
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
  ret void
}

define spir_kernel void @bar(ptr addrspace(1) %_arg_GlobalA, ptr byval(%id) %_arg_GlobalId, ptr addrspace(3) %_arg_LocalA) {
entry:
  %var = alloca i32
  %p0 = load i64, ptr %_arg_GlobalId
  %add = getelementptr inbounds i32, ptr addrspace(1) %_arg_GlobalA, i64 %p0
  %p2 = load i64, ptr addrspace(1) @__spirv_BuiltInGlobalInvocationId
  %idx = getelementptr inbounds i32, ptr addrspace(1) %add, i64 %p2
  %var1 = addrspacecast ptr addrspace(1) %idx to ptr addrspace(4)
  %var2 = addrspacecast ptr addrspace(3) %_arg_LocalA to ptr addrspace(4)
  %var3 = addrspacecast ptr %var to ptr addrspace(4)
  %G = call spir_func ptr addrspace(1) @_Z9to_globalPv(ptr addrspace(4) %var1, i32 5)
  %L = call spir_func ptr addrspace(3) @_Z8to_localPv(ptr addrspace(4) %var2, i32 4)
  %P = call spir_func ptr @_Z10to_privatePv(ptr addrspace(4) %var3, i32 7)
  ret void
}

declare spir_func ptr addrspace(1) @_Z33__spirv_GenericCastToPtr_ToGlobalPvi(ptr addrspace(4), i32)
declare spir_func ptr addrspace(3) @_Z32__spirv_GenericCastToPtr_ToLocalPvi(ptr addrspace(4), i32)
declare spir_func ptr @_Z34__spirv_GenericCastToPtr_ToPrivatePvi(ptr addrspace(4), i32)

declare spir_func ptr addrspace(1) @_Z9to_globalPv(ptr addrspace(4))
declare spir_func ptr addrspace(3) @_Z8to_localPv(ptr addrspace(4))
declare spir_func ptr @_Z10to_privatePv(ptr addrspace(4))

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#FloatTy:]] = OpTypeFloat 32
; CHECK-DAG: %[[#StructTy:]] = OpTypeStruct %[[#FloatTy]]
; CHECK-DAG: %[[#ArrayTy:]] = OpTypeArray %[[#StructTy]] %[[#]]
; CHECK-DAG: %[[#Struct7Ty:]] = OpTypeStruct %[[#ArrayTy]]
; CHECK-DAG: %[[#Void:]] = OpTypeVoid
; CHECK-DAG: %[[#PtrStructTy:]] = OpTypePointer Generic %[[#StructTy]]
; CHECK-DAG: %[[#PtrStruct7Ty:]] = OpTypePointer Generic %[[#Struct7Ty]]
; CHECK-DAG: %[[#FunTy:]] = OpTypeFunction %[[#Void]] %[[#PtrStructTy]] %[[#PtrStruct7Ty]]
; CHECK-DAG: %[[#Const1:]] = OpConstant %[[#FloatTy]] 1
; CHECK-DAG: %[[#FPtrStructTy:]] = OpTypePointer Function %[[#StructTy]]
; CHECK-DAG: %[[#Spec1:]] = OpSpecConstantComposite %[[#StructTy]] %[[#Const1]]
; CHECK-DAG: %[[#Spec2:]] = OpSpecConstantComposite %[[#ArrayTy]] %[[#Spec1]] %[[#Spec1]]
; CHECK-DAG: %[[#Spec3:]] = OpSpecConstantComposite %[[#Struct7Ty]] %[[#Spec2]]
; CHECK: %[[#FunDef:]] = OpFunction %[[#Void]] None %[[#FunTy]]
; CHECK: %[[#Arg1:]] = OpFunctionParameter %[[#PtrStructTy]]
; CHECK: %[[#Arg2:]] = OpFunctionParameter %[[#PtrStruct7Ty]]
; CHECK: %[[#]] = OpVariable %[[#FPtrStructTy]] Function
; CHECK: OpStore %[[#Arg1]] %[[#Spec1]]
; CHECK: OpStore %[[#Arg2]] %[[#Spec3]]
; CHECK: OpFunctionEnd

%Struct = type <{ float }>
%Struct7 = type [2 x %Struct]
%Nested = type { %Struct7 }

define spir_kernel void @foo(ptr addrspace(4) %arg1, ptr addrspace(4) %arg2) {
entry:
  %var = alloca %Struct
  %r1 = call %Struct @_Z29__spirv_SpecConstantComposite_1(float 1.0)
  store %Struct %r1, ptr addrspace(4) %arg1
  %r2 = call %Struct7 @_Z29__spirv_SpecConstantComposite_2(%Struct %r1, %Struct %r1)
  %r3 = call %Nested @_Z29__spirv_SpecConstantComposite_3(%Struct7 %r2)
  store %Nested %r3, ptr addrspace(4) %arg2

  ret void
}

declare %Struct @_Z29__spirv_SpecConstantComposite_1(float)
declare %Struct7 @_Z29__spirv_SpecConstantComposite_2(%Struct, %Struct)
declare %Nested @_Z29__spirv_SpecConstantComposite_3(%Struct7)

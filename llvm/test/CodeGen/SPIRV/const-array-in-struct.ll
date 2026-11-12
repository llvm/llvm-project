; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-DAG: %[[#IntTy:]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: %[[#Const16:]] = OpConstant %[[#IntTy]] 8
; CHECK-SPIRV-DAG: %[[#ArrayTy:]] = OpTypeArray %[[#IntTy]] %[[#Const16]]
; CHECK-SPIRV-DAG: %[[#StructTy:]] = OpTypeStruct %[[#ArrayTy]]
; CHECK-SPIRV-DAG: %[[#]] = OpConstantNull %[[#IntTy]]
; CHECK-SPIRV-DAG: %[[#Const1:]] = OpConstant %[[#IntTy]] 1
; CHECK-SPIRV-DAG: %[[#Const2:]] = OpConstant %[[#IntTy]] 2
; CHECK-SPIRV-DAG: %[[#Const3:]] = OpConstant %[[#IntTy]] 3
; CHECK-SPIRV-DAG: %[[#Const4:]] = OpConstant %[[#IntTy]] 4
; CHECK-SPIRV-DAG: %[[#Const5:]] = OpConstant %[[#IntTy]] 5
; CHECK-SPIRV-DAG: %[[#Const6:]] = OpConstant %[[#IntTy]] 6
; CHECK-SPIRV-DAG: %[[#Const7:]] = OpConstant %[[#IntTy]] 7
; CHECK-SPIRV-DAG: %[[#ConstArray:]] = OpConstantComposite %[[#ArrayTy]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#]]
; CHECK-SPIRV-DAG: %[[#]] = OpConstantComposite %[[#StructTy]] %[[#ConstArray]]

%struct_array_8i32 = type { [8 x i32] }

@G = private unnamed_addr addrspace(1) constant %struct_array_8i32 { [8 x i32] [i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7] }, align 4

define spir_kernel void @test() {
  ret void
}

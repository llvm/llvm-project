; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV: %[[#IntTy:]] = OpTypeInt 32 0
; CHECK-SPIRV: %[[#Const16:]] = OpConstant %[[#IntTy]] 16
; CHECK-SPIRV: %[[#ArrayTy:]] = OpTypeArray %[[#IntTy]] %[[#Const16]]
; CHECK-SPIRV: %[[#StructTy:]] = OpTypeStruct %[[#ArrayTy]]
; CHECK-SPIRV-COUNT-16: %[[#]] = OpConstant %[[#IntTy]] {{[0-9]+}}
; CHECK-SPIRV: %[[#ConstArray:]] = OpConstantComposite %[[#ArrayTy]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#]]
; CHECK-SPIRV: %[[#]] = OpConstantComposite %[[#StructTy]] %[[#ConstArray]]

%struct_array_16i32 = type { [16 x i32] }

@G = private unnamed_addr addrspace(1) constant %struct_array_16i32 { [16 x i32] [i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15] }, align 4

define spir_kernel void @test() {
  ret void
}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG:  %[[#uint_ty:]] = OpTypeInt 32 0
; CHECK-DAG:   %[[#uint_0:]] = OpConstant %[[#uint_ty]] 0{{$}}
; CHECK-DAG:   %[[#int_10:]] = OpConstant %[[#uint_ty]] 10{{$}}
; CHECK-DAG: %[[#array_ty:]] = OpTypeArray %[[#uint_ty]] %[[#int_10]]
; CHECK-DAG: %[[#array_fp:]] = OpTypePointer Function %[[#array_ty]]
; CHECK-DAG: %[[#array_pp:]] = OpTypePointer Private %[[#array_ty]]
; CHECK-DAG:   %[[#int_fp:]] = OpTypePointer Function %[[#uint_ty]]
; CHECK-DAG:   %[[#int_pp:]] = OpTypePointer Private %[[#uint_ty]]

@gv = external addrspace(10) global [10 x i32]
; CHECK: %[[#gv:]] = OpVariable %[[#array_pp]] Private

define internal spir_func i32 @foo() {
  %array = alloca [10 x i32], align 4
; CHECK: %[[#array:]] = OpVariable %[[#array_fp:]] Function

  ; Direct load from the pointer index. This requires an OpAccessChain
  %1 = load i32, ptr %array, align 4
; CHECK: %[[#ptr:]] = OpAccessChain %[[#int_fp]] %[[#array]] %[[#uint_0]]
; CHECK: %[[#val:]] = OpLoad %[[#uint_ty]] %[[#ptr]] Aligned 4

  ret i32 %1
; CHECK: OpReturnValue %[[#val]]
}

define internal spir_func i32 @bar() {
  ; Direct load from the pointer index. This requires an OpAccessChain
  %1 = load i32, ptr addrspace(10) @gv
; CHECK: %[[#ptr:]] = OpAccessChain %[[#int_pp]] %[[#gv]] %[[#uint_0]]
; CHECK: %[[#val:]] = OpLoad %[[#uint_ty]] %[[#ptr]] Aligned 4

  ret i32 %1
; CHECK: OpReturnValue %[[#val]]
}

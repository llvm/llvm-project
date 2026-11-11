; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-DAG: %[[#Int32:]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: %[[#Bool:]] = OpTypeBool
; CHECK-SPIRV-DAG: %[[#V8Int32:]] = OpTypeVector %[[#Int32]] 8
; CHECK-SPIRV-DAG: %[[#V8Bool:]] = OpTypeVector %[[#Bool]] 8
; CHECK-SPIRV-DAG: %[[#One:]] = OpConstant %[[#Int32]] 1
; CHECK-SPIRV-DAG: %[[#Ones:]] = OpConstantComposite %[[#V8Int32]] %[[#One]] %[[#One]] %[[#One]] %[[#One]] %[[#One]] %[[#One]] %[[#One]] %[[#One]]
; CHECK-SPIRV-DAG: %[[#Zeros:]] = OpConstantNull %[[#V8Int32]]

; CHECK-SPIRV:      OpBitwiseAnd
; CHECK-SPIRV-NEXT: OpINotEqual

define spir_kernel void @test(i32 %op1, i32 %op2, i8 %op3, ptr addrspace(1) %out) {
entry:
  %0 = trunc i8 %op3 to i1
  %call = call spir_func i32 @_Z14__spirv_Selectbii(i1 zeroext %0, i32 %op1, i32 %op2)
  store i32 %call, ptr addrspace(1) %out
  ret void
}

; The <8 x i32> bitwise and result is wider than 128 bits.
; CHECK-SPIRV:      %[[#Op1:]] = OpFunctionParameter %[[#V8Int32]]
; CHECK-SPIRV:      %[[#And:]] = OpBitwiseAnd %[[#V8Int32]] %[[#Op1]] %[[#Ones]]
; CHECK-SPIRV-NEXT: OpINotEqual %[[#V8Bool]] %[[#And]] %[[#Zeros]]
define spir_kernel void @test_v8(<8 x i32> %op1, <8 x i32> %op2, ptr addrspace(1) %out) {
entry:
  %0 = trunc <8 x i32> %op1 to <8 x i1>
  %r = select <8 x i1> %0, <8 x i32> %op1, <8 x i32> %op2
  store <8 x i32> %r, ptr addrspace(1) %out
  ret void
}

declare spir_func i32 @_Z14__spirv_Selectbii(i1 zeroext, i32, i32)

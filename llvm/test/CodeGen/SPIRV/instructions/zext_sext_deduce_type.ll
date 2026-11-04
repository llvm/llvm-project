; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[#I8:]] = OpTypeInt 8
; CHECK: %[[#I64:]] = OpTypeInt 64
; CHECK: %[[#UINT32_MAX:]] = OpConstant %[[#I64]] 4294967295
; CHECK: %[[#SHIFT:]] = OpConstant %[[#I64]] 2097152
; CHECK-DAG: %[[#X:]] = OpFunctionParameter %[[#I8]]
; CHECK: %[[#Y:]] = OpFunctionParameter %[[#I64]]
; CHECK-DAG: %[[#SEXT:]] = OpSConvert %[[#I64]] %[[#X]]
; CHECK: %[[#AND:]] = OpBitwiseAnd %[[#I64]] %[[#SEXT]] %[[#UINT32_MAX]]
; CHECK: %[[#]] = OpShiftRightArithmetic %[[#I64]] %[[#SHIFT]] %[[#AND]]

define i64 @foo(i8 %x, i64 %y) {
  %2 = sext i8 %x to i32
  %3 = zext i32 %2 to i64
  %4 = ashr i64 2097152, %3

  ret i64 %4
}
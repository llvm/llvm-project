; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: opt -S -passes='spirv-emit-intrinsics,function(spirv-legalize-pointer-cast)' -mtriple=spirv-unknown-unknown %s | FileCheck %s --check-prefix=IR
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#I16:]] = OpTypeInt 16 0
; CHECK-DAG: %[[#I64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#V4I16:]] = OpTypeVector %[[#I16]] 4
; CHECK-DAG: %[[#PTR4:]] = OpTypePointer Function %[[#V4I16]]

define i64 @test(i64 %value) {
; CHECK-LABEL: ; -- Begin function test
; CHECK: %[[#VALUE:]] = OpFunctionParameter %[[#I64]]
; CHECK: %[[#PTR:]] = OpVariable %[[#PTR4]] Function
; CHECK: %[[#STORE:]] = OpBitcast %[[#V4I16]] %[[#VALUE]]
; CHECK: OpStore %[[#PTR]] %[[#STORE]]
; CHECK: %[[#LOAD:]] = OpLoad %[[#V4I16]] %[[#PTR]]
; CHECK: %[[#SCALAR:]] = OpBitcast %[[#I64]] %[[#LOAD]]
; CHECK: OpReturnValue %[[#SCALAR]]
; IR-LABEL: define i64 @test(
; IR: call <4 x i16> @llvm.spv.bitcast.v4i16.i64(i64 %value)
; IR: store <4 x i16> %{{.*}}, ptr %ptr, align 2
; IR: load <4 x i16>, ptr %ptr, align 2
; IR: call i64 @llvm.spv.bitcast.i64.v4i16(
entry:
  %ptr = alloca <4 x i16>, align 2
  store i64 %value, ptr %ptr, align 2
  %loaded = load i64, ptr %ptr, align 2
  ret i64 %loaded
}

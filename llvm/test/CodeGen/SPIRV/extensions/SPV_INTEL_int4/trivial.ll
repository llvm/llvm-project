; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_int4 %s -o - | FileCheck %s
; RUNx: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_int4 %s -o - -filetype=obj | spirv-val %}

; CHECK: Capability Int4TypeINTEL
; CHECK: Extension "SPV_INTEL_int4"
; CHECK: %[[#Int4:]] = OpTypeInt  4 0
; CHECK: OpTypeFunction %[[#]] %[[#Int4]]
; CHECK: %[[#Int4PtrTy:]] = OpTypePointer Function %[[#Int4]]
; CHECK: %[[#Const:]] = OpConstant %[[#Int4]]  1

; CHECK: %[[#Int4Ptr:]] = OpVariable %[[#Int4PtrTy]] Function
; CHECK: OpStore %[[#Int4Ptr]] %[[#Const]]
; CHECK: %[[#Load:]] = OpLoad %[[#Int4]] %[[#Int4Ptr]]
; CHECK: OpFunctionCall %[[#]] %[[#]] %[[#Load]]

define spir_kernel void @foo() {
entry:
  %0 = alloca i4
  store i4 1, ptr %0
  %1 = load i4, ptr %0
  call spir_func void @boo(i4 %1)
  ret void
}

declare spir_func void @boo(i4)

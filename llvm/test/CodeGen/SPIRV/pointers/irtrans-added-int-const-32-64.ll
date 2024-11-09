; The goal of the test is to check that collisions between explicit integer constants and
; integer constants added automatically by IR Translator are resolved correctly both for
; 32- and 64-bits platforms. The test is successful if it doesn't crash and generate valid
; SPIR-V code for both 32 and 64 bits targets.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV64
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV32
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV64-DAG: %[[#IntTy:]] = OpTypeInt 64 0
; CHECK-SPIRV32-DAG: %[[#IntTy:]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: %[[#Const2:]] = OpConstant %[[#IntTy]] 2
; CHECK-SPIRV-DAG: %[[#]] = OpSpecConstantOp %[[#]] 70 %[[#]] %[[#]] %[[#Const2]]
; CHECK-SPIRV: OpFunction

@a_var = addrspace(1) global [2 x i8] [i8 1, i8 1]
@p_var = addrspace(1) global ptr addrspace(1) getelementptr (i8, ptr addrspace(1) @a_var, i64 2)

define spir_func void @foo() {
entry:
  ret void
}

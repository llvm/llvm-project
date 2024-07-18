; This test case ensures that cleaning of temporary constants doesn't purge tracked ones.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV: %[[#Int:]] = OpTypeInt 32 0
; CHECK-SPIRV: %[[#C0:]] = OpConstant %[[#Int]] 0
; CHECK-SPIRV: %[[#C1:]] = OpConstant %[[#Int]] 1
; CHECK-SPIRV: OpSelect %[[#Int]] %[[#]] %[[#C1]] %[[#C0]]


define spir_kernel void @foo() {
entry:
  %addr = alloca i32
  %r1 = call i8 @_Z20__spirv_SpecConstantia(i32 0, i8 1)
  ; The name '%conv17.i' is important for the test case,
  ; because it includes i32 0 when encoded for SPIR-V usage.
  %conv17.i = sext i8 %r1 to i64
  %tobool = trunc i8 %r1 to i1
  %r2 = zext i1 %tobool to i32
  store i32 %r2, ptr %addr
  ret void
}

declare i8 @_Z20__spirv_SpecConstantia(i32, i8)

; This test ensures that LLVM IR bitwise instructions result in logical SPIR-V instructions
; when applied to i1 type

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#Char:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#Vec2Char:]] = OpTypeVector %[[#Char]] 2
; CHECK-DAG: %[[#Bool:]] = OpTypeBool
; CHECK-DAG: %[[#Vec2Bool:]] = OpTypeVector %[[#Bool]] 2

; CHECK: OpBitwiseAnd %[[#Char]]
; CHECK: OpBitwiseOr %[[#Char]]
; CHECK: OpBitwiseXor %[[#Char]]
; CHECK: OpBitwiseAnd %[[#Vec2Char]]
; CHECK: OpBitwiseOr %[[#Vec2Char]]
; CHECK: OpBitwiseXor %[[#Vec2Char]]

; CHECK: OpLogicalAnd %[[#Bool]]

; CHECK: OpLogicalAnd %[[#Bool]]
; CHECK: OpLogicalOr %[[#Bool]]
; CHECK: OpLogicalNotEqual %[[#Bool]]
; CHECK: OpLogicalAnd %[[#Vec2Bool]]
; CHECK: OpLogicalOr %[[#Vec2Bool]]
; CHECK: OpLogicalNotEqual %[[#Vec2Bool]]

define void @test1(i8 noundef %arg1, i8 noundef %arg2) {
  %cond1 = and i8 %arg1, %arg2
  %cond2 = or i8 %arg1, %arg2
  %cond3 = xor i8 %arg1, %arg2
  ret void
}

define void @test1v(<2 x i8> noundef %arg1, <2 x i8> noundef %arg2) {
  %cond1 = and <2 x i8> %arg1, %arg2
  %cond2 = or <2 x i8> %arg1, %arg2
  %cond3 = xor <2 x i8> %arg1, %arg2
  ret void
}

define void @test2(float noundef %real, float noundef %imag) {
entry:
  %realabs = tail call spir_func noundef float @_Z16__spirv_ocl_fabsf(float noundef %real)
  %cond1 = fcmp oeq float %realabs, 1.000000e+00
  %cond2 = fcmp oeq float %imag, 0.000000e+00
  %cond3 = and i1 %cond1, %cond2
  br i1 %cond3, label %midlbl, label %cleanup
midlbl:
  br label %cleanup
cleanup:
  ret void
}

define void @test3(i1 noundef %arg1, i1 noundef %arg2) {
  %cond1 = and i1 %arg1, %arg2
  %cond2 = or i1 %arg1, %arg2
  %cond3 = xor i1 %arg1, %arg2
  ret void
}

define void @test3v(<2 x i1> noundef %arg1, <2 x i1> noundef %arg2) {
  %cond1 = and <2 x i1> %arg1, %arg2
  %cond2 = or <2 x i1> %arg1, %arg2
  %cond3 = xor <2 x i1> %arg1, %arg2
  ret void
}

declare dso_local spir_func noundef float @_Z16__spirv_ocl_fabsf(float noundef)

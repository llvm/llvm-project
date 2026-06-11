; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-DAG: OpDecorate %[[#ZEXT:]] FuncParamAttr Zext
; CHECK-SPIRV-DAG: OpDecorate %[[#SEXT:]] FuncParamAttr Sext
; CHECK-SPIRV-DAG: OpDecorate %[[#NOWRITE:]] FuncParamAttr NoWrite
; CHECK-SPIRV-DAG: OpDecorate %[[#NOALIAS:]] FuncParamAttr NoAlias
; CHECK-SPIRV-DAG: OpDecorate %[[#BYVAL:]] FuncParamAttr ByVal
; CHECK-SPIRV-DAG: OpDecorate %[[#SRET:]] FuncParamAttr Sret

; CHECK-SPIRV: %[[#ZEXT]] = OpFunctionParameter %[[#]]
define spir_func void @test_zext(i8 zeroext %arg) {
entry:
  ret void
}

; CHECK-SPIRV: %[[#SEXT]] = OpFunctionParameter %[[#]]
define spir_func void @test_sext(i8 signext %arg) {
entry:
  ret void
}

; CHECK-SPIRV: %[[#NOWRITE]] = OpFunctionParameter %[[#]]
define spir_func void @test_readonly(ptr readonly %arg) {
entry:
  ret void
}

; CHECK-SPIRV: %[[#NOALIAS]] = OpFunctionParameter %[[#]]
define spir_func void @test_noalias(ptr noalias %arg) {
entry:
  ret void
}

; CHECK-SPIRV: %[[#BYVAL]] = OpFunctionParameter %[[#]]
define spir_func void @test_byval(ptr byval(i32) %arg) {
entry:
  ret void
}

; CHECK-SPIRV: %[[#SRET]] = OpFunctionParameter %[[#]]
define spir_func void @test_sret(ptr sret(i32) %arg) {
entry:
  ret void
}

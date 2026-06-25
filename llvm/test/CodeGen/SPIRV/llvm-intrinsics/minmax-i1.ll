; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_ALTERA_arbitrary_precision_integers %s -o - | FileCheck %s

; CHECK-DAG: %[[#BOOL:]] = OpTypeBool
; CHECK-DAG: %[[#INT:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#VBOOL:]] = OpTypeVector %[[#BOOL]] 2
; CHECK-DAG: %[[#ZERO:]] = OpConstantNull %[[#INT]]
; CHECK-DAG: %[[#ONE:]] = OpConstant %[[#INT]] 1

define spir_func i1 @test_umin_i1(i1 %a, i1 %b) {
entry:
; CHECK: OpSelect %[[#INT]] %[[#]] %[[#ONE]] %[[#ZERO]]
; CHECK: OpSelect %[[#INT]] %[[#]] %[[#ONE]] %[[#ZERO]]
; CHECK: OpExtInst %[[#INT]] %[[#]] u_min %[[#]] %[[#]]
; CHECK: OpBitwiseAnd %[[#INT]] %[[#]] %[[#ONE]]
; CHECK: OpINotEqual %[[#BOOL]] %[[#]] %[[#ZERO]]
  %r = call i1 @llvm.umin.i1(i1 %a, i1 %b)
  ret i1 %r
}

define spir_func i1 @test_umax_i1(i1 %a, i1 %b) {
entry:
; CHECK: OpExtInst %[[#INT]] %[[#]] u_max %[[#]] %[[#]]
; CHECK: OpINotEqual %[[#BOOL]] %[[#]] %[[#ZERO]]
  %r = call i1 @llvm.umax.i1(i1 %a, i1 %b)
  ret i1 %r
}

define spir_func i1 @test_smin_i1(i1 %a, i1 %b) {
entry:
; CHECK: OpExtInst %[[#INT]] %[[#]] s_min %[[#]] %[[#]]
; CHECK: OpINotEqual %[[#BOOL]] %[[#]] %[[#ZERO]]
  %r = call i1 @llvm.smin.i1(i1 %a, i1 %b)
  ret i1 %r
}

define spir_func i1 @test_smax_i1(i1 %a, i1 %b) {
entry:
; CHECK: OpExtInst %[[#INT]] %[[#]] s_max %[[#]] %[[#]]
; CHECK: OpINotEqual %[[#BOOL]] %[[#]] %[[#ZERO]]
  %r = call i1 @llvm.smax.i1(i1 %a, i1 %b)
  ret i1 %r
}

define spir_func <2 x i1> @test_umin_v2i1(<2 x i1> %a, <2 x i1> %b) {
entry:
; CHECK: OpCompositeExtract %[[#BOOL]] %[[#]] 0
; CHECK: OpCompositeExtract %[[#BOOL]] %[[#]] 1
; CHECK: OpCompositeExtract %[[#BOOL]] %[[#]] 0
; CHECK: OpCompositeExtract %[[#BOOL]] %[[#]] 1
; CHECK: OpExtInst %[[#INT]] %[[#]] u_min %[[#]] %[[#]]
; CHECK: OpINotEqual %[[#BOOL]] %[[#]] %[[#ZERO]]
; CHECK: OpExtInst %[[#INT]] %[[#]] u_min %[[#]] %[[#]]
; CHECK: OpINotEqual %[[#BOOL]] %[[#]] %[[#ZERO]]
; CHECK: OpCompositeConstruct %[[#VBOOL]] %[[#]] %[[#]]
  %r = call <2 x i1> @llvm.umin.v2i1(<2 x i1> %a, <2 x i1> %b)
  ret <2 x i1> %r
}

declare i1 @llvm.umin.i1(i1, i1)
declare i1 @llvm.umax.i1(i1, i1)
declare i1 @llvm.smin.i1(i1, i1)
declare i1 @llvm.smax.i1(i1, i1)
declare <2 x i1> @llvm.umin.v2i1(<2 x i1>, <2 x i1>)

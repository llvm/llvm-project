; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_long_vector %s -o - | FileCheck %s
; spirv-val seems to have problems reading OpTypeVectorIdEXT correctly, enable once fixed
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_long_vector %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Vec17:]] = OpTypeVectorIdEXT %[[#Int]] 17
; CHECK-DAG: %[[#PtrGlobalInt:]] = OpTypePointer CrossWorkgroup %[[#Int]]
; CHECK-DAG: %[[#Const10:]] = OpConstant %[[#Int]] 10
; CHECK-DAG: %[[#Const20:]] = OpConstant %[[#Int]] 20
; CHECK-DAG: %[[#Const30:]] = OpConstant %[[#Int]] 30
; CHECK-DAG: %[[#Const40:]] = OpConstant %[[#Int]] 40
; CHECK-DAG: %[[#Const50:]] = OpConstant %[[#Int]] 50
; CHECK-DAG: %[[#Const60:]] = OpConstant %[[#Int]] 60

@G = internal addrspace(1) global i32 0, align 4

define void @main() {
entry:
; CHECK: %[[#Var:]] = OpVariable %[[#PtrGlobalInt]] CrossWorkgroup

; CHECK: %[[#Idx:]] = OpLoad %[[#Int]]
  %idx = load i32, ptr addrspace(1) @G, align 4


; CHECK: OpCompositeInsert %[[#Vec17]] %[[#Const10]]
  %vec = insertelement <17 x i32> poison, i32 10, i64 0

; CHECK: OpCompositeInsert %[[#Vec17]] %[[#Const20]]
  %vec2 = insertelement <17 x i32> %vec, i32 20, i64 1

; CHECK: OpCompositeInsert %[[#Vec17]] %[[#Const30]]
  %vec3 = insertelement <17 x i32> %vec2, i32 30, i64 2

; CHECK: OpCompositeInsert %[[#Vec17]] %[[#Const40]]
  %vec4 = insertelement <17 x i32> %vec3, i32 40, i64 3

; CHECK: OpCompositeInsert %[[#Vec17]] %[[#Const50]]
  %vec5 = insertelement <17 x i32> %vec4, i32 50, i64 4

; CHECK: %[[#V6:]] = OpCompositeInsert %[[#Vec17]] %[[#Const60]]
  %vec6 = insertelement <17 x i32> %vec5, i32 60, i64 5

; CHECK: %[[#Res:]] = OpVectorExtractDynamic %[[#Int]] %[[#V6]] %[[#Idx]]
  %res = extractelement <17 x i32> %vec6, i32 %idx

; CHECK: OpStore {{.*}} %[[#Res]]
  store i32 %res, ptr addrspace(1) @G, align 4
  ret void
}
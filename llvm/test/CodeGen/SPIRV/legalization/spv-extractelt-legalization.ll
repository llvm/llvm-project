; RUN: llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Const0:]] = OpConstant %[[#Int]] 0
; CHECK-DAG: %[[#Const1:]] = OpConstant %[[#Int]] 1
; CHECK-DAG: %[[#Const2:]] = OpConstant %[[#Int]] 2
; CHECK-DAG: %[[#Const3:]] = OpConstant %[[#Int]] 3
; CHECK-DAG: %[[#Const4:]] = OpConstant %[[#Int]] 4
; CHECK-DAG: %[[#Const5:]] = OpConstant %[[#Int]] 5
; CHECK-DAG: %[[#Const10:]] = OpConstant %[[#Int]] 10
; CHECK-DAG: %[[#Const20:]] = OpConstant %[[#Int]] 20
; CHECK-DAG: %[[#Const30:]] = OpConstant %[[#Int]] 30
; CHECK-DAG: %[[#Const40:]] = OpConstant %[[#Int]] 40
; CHECK-DAG: %[[#Const50:]] = OpConstant %[[#Int]] 50
; CHECK-DAG: %[[#Const60:]] = OpConstant %[[#Int]] 60
; CHECK-DAG: %[[#Arr:]] = OpTypeArray %[[#Int]] %[[#]]
; CHECK-DAG: %[[#PtrArr:]] = OpTypePointer Function %[[#Arr]]
; CHECK-DAG: %[[#PtrPriv:]] = OpTypePointer Private %[[#Int]]

@G = internal addrspace(10) global i32 0, align 4

define void @main() #0 {
entry:
; CHECK: %[[#Var:]] = OpVariable %[[#PtrArr]] Function

; CHECK: %[[#Idx:]] = OpLoad %[[#Int]]
  %idx = load i32, ptr addrspace(10) @G, align 4


; CHECK: %[[#PtrElt0:]] = OpInBoundsAccessChain %[[#]] %[[#Var]] %[[#Const0]]
; CHECK: OpStore %[[#PtrElt0]] %[[#Const10]]
  %vec = insertelement <6 x i32> poison, i32 10, i64 0

; CHECK: %[[#PtrElt1:]] = OpInBoundsAccessChain %[[#]] %[[#Var]] %[[#Const1]]
; CHECK: OpStore %[[#PtrElt1]] %[[#Const20]]
  %vec2 = insertelement <6 x i32> %vec, i32 20, i64 1

; CHECK: %[[#PtrElt2:]] = OpInBoundsAccessChain %[[#]] %[[#Var]] %[[#Const2]]
; CHECK: OpStore %[[#PtrElt2]] %[[#Const30]]
  %vec3 = insertelement <6 x i32> %vec2, i32 30, i64 2

; CHECK: %[[#PtrElt3:]] = OpInBoundsAccessChain %[[#]] %[[#Var]] %[[#Const3]]
; CHECK: OpStore %[[#PtrElt3]] %[[#Const40]]
  %vec4 = insertelement <6 x i32> %vec3, i32 40, i64 3

; CHECK: %[[#PtrElt4:]] = OpInBoundsAccessChain %[[#]] %[[#Var]] %[[#Const4]]
; CHECK: OpStore %[[#PtrElt4]] %[[#Const50]]
  %vec5 = insertelement <6 x i32> %vec4, i32 50, i64 4

; CHECK: %[[#PtrElt5:]] = OpInBoundsAccessChain %[[#]] %[[#Var]] %[[#Const5]]
; CHECK: OpStore %[[#PtrElt5]] %[[#Const60]]
  %vec6 = insertelement <6 x i32> %vec5, i32 60, i64 5

; CHECK: %[[#Ptr:]] = OpInBoundsAccessChain %[[#]] %[[#Var]] %[[#Idx]]
; CHECK: %[[#Ld:]] = OpLoad %[[#Int]] %[[#Ptr]]
  %res = extractelement <6 x i32> %vec6, i32 %idx
  
; CHECK: OpStore {{.*}} %[[#Ld]]
  store i32 %res, ptr addrspace(10) @G, align 4
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
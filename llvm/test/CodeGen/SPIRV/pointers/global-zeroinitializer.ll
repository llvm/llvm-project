; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpName %[[#Var:]] "var"
; CHECK-DAG: %[[#Char:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#Vec2Char:]] = OpTypeVector %[[#Char]] 2
; CHECK-DAG: %[[#PtrVec2Char:]] = OpTypePointer CrossWorkgroup %[[#Vec2Char]]
; CHECK-DAG: %[[#ConstNull:]] = OpConstantNull %[[#Vec2Char]]
; CHECK: %[[#]] = OpVariable %[[#PtrVec2Char]] CrossWorkgroup %[[#ConstNull]]
; As an option: %[[#C0:]] = OpConstant %[[#Char]] 0
;               %[[#VecZero:]] = OpConstantComposite %[[#Vec2Char]] %[[#C0]] %[[#C0]]
;               %[[#]] = OpVariable %[[#PtrVec2Char]] CrossWorkgroup %[[#VecZero]]
; CHECK: OpFunction

@var = addrspace(1) global <2 x i8> zeroinitializer
;@var = addrspace(1) global <2 x i8> <i8 1, i8 1>

define spir_kernel void @foo() {
entry:
  %addr = load <2 x i8>, ptr addrspace(1) @var, align 2
  ret void
}

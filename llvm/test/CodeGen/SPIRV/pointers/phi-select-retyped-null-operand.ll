; A PHI or select that is given a new element type after its operands were
; validated must cast its null operands to the new type. %b is first typed as
; i8 and only later, through %a.be, as float. The predecessor order of %header
; decides the order in which LoopSimplify creates the blocks, and thereby the
; order in which the PHIs are visited.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#Char:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#CharPtr:]] = OpTypePointer Function %[[#Char]]
; CHECK-DAG: %[[#Float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#FloatPtr:]] = OpTypePointer Function %[[#Float]]
; CHECK-DAG: %[[#Null:]] = OpConstantNull %[[#CharPtr]]

; CHECK-LABEL: ; -- Begin function phi
; CHECK: %[[#PhiNull1:]] = OpBitcast %[[#FloatPtr]] %[[#Null]]
; CHECK: %[[#PhiNull2:]] = OpBitcast %[[#FloatPtr]] %[[#Null]]
; CHECK: %[[#PhiNull3:]] = OpBitcast %[[#FloatPtr]] %[[#Null]]
; CHECK: OpPhi %[[#FloatPtr]] %[[#PhiNull3]] %[[#]]
; CHECK: OpPhi %[[#FloatPtr]] %[[#PhiNull1]] %[[#]]
; CHECK: OpPhi %[[#FloatPtr]] %[[#PhiNull2]] %[[#]]

; CHECK-LABEL: ; -- Begin function select
; CHECK: %[[#SelNull1:]] = OpBitcast %[[#FloatPtr]] %[[#Null]]
; CHECK: %[[#SelNull2:]] = OpBitcast %[[#FloatPtr]] %[[#Null]]
; CHECK: %[[#SelNull3:]] = OpBitcast %[[#FloatPtr]] %[[#Null]]
; CHECK: OpPhi %[[#FloatPtr]] %[[#SelNull3]] %[[#]]
; CHECK: OpSelect %[[#FloatPtr]] %[[#]] %[[#SelNull1]] %[[#]]
; CHECK: OpPhi %[[#FloatPtr]] %[[#SelNull2]] %[[#]]

define spir_kernel void @phi(i1 %c, ptr %p) {
entry:
  br label %header

header:
  %a = phi ptr [ null, %entry ], [ null, %body ], [ %b, %latch ]
  br i1 %c, label %cond, label %latch

cond:
  br i1 %c, label %latch, label %body

body:
  %v = load float, ptr %a, align 4
  br i1 %c, label %exit, label %header

latch:
  %b = phi ptr [ null, %cond ], [ %p, %header ]
  %dead = icmp eq i64 0, 0
  br label %header

exit:
  store float %v, ptr addrspace(3) null
  ret void

  uselistorder label %header, { 1, 0, 2 }
}

define spir_kernel void @select(i1 %c, ptr %p) {
entry:
  br label %header

header:
  %a = phi ptr [ null, %entry ], [ %a.be, %header.backedge ]
  br i1 %c, label %cond, label %latch

cond:
  br i1 %c, label %latch, label %body

body:
  %v = load float, ptr %a, align 4
  br i1 %c, label %exit, label %header.backedge

latch:
  %b = select i1 %c, ptr null, ptr %p
  %dead = icmp eq i64 0, 0
  br label %header.backedge

header.backedge:
  %a.be = phi ptr [ null, %body ], [ %b, %latch ]
  br label %header

exit:
  store float %v, ptr addrspace(3) null
  ret void
}

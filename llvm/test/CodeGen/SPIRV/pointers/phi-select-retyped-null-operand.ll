; A PHI or select that is given a new element type after its operands were
; validated must cast its null operands to the new type. %b is first typed as
; i8 and only later, through %a.be, as float. The IR is already loop-simplified
; so that LoopSimplify leaves the block order alone.

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
  %a = phi ptr [ null, %entry ], [ %a.be, %backedge ]
  br i1 %c, label %cond, label %join

cond:
  br i1 %c, label %join, label %body

body:
  %v = load float, ptr %a, align 4
  br i1 %c, label %exit, label %backedge

join:
  %b = phi ptr [ null, %cond ], [ %p, %header ]
  br label %backedge

backedge:
  %a.be = phi ptr [ null, %body ], [ %b, %join ]
  br label %header

exit:
  store float %v, ptr addrspace(3) null
  ret void
}

define spir_kernel void @select(i1 %c, ptr %p) {
entry:
  br label %header

header:
  %a = phi ptr [ null, %entry ], [ %a.be, %backedge ]
  br i1 %c, label %cond, label %join

cond:
  br i1 %c, label %join, label %body

body:
  %v = load float, ptr %a, align 4
  br i1 %c, label %exit, label %backedge

join:
  %b = select i1 %c, ptr null, ptr %p
  br label %backedge

backedge:
  %a.be = phi ptr [ null, %body ], [ %b, %join ]
  br label %header

exit:
  store float %v, ptr addrspace(3) null
  ret void
}

; A constant holding @g (i32 pointee) is rebuilt with OpCompositeInsert in its
; LLVM type for any user other than an intrinsic or extractvalue.

; RUN: llc -O0 --spirv-ext=+SPV_INTEL_masked_gather_scatter -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 --spirv-ext=+SPV_INTEL_masked_gather_scatter -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Char:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#CharPtr:]] = OpTypePointer CrossWorkgroup %[[#Char]]
; CHECK-DAG: %[[#Q:]] = OpTypeStruct %[[#CharPtr]] %[[#Int]]
; CHECK-DAG: %[[#Vec:]] = OpTypeVector %[[#CharPtr]] 2
; CHECK-DAG: %[[#R:]] = OpTypeStruct %[[#Q]] %[[#Int]]
; CHECK-DAG: %[[#G:]] = OpVariable %[[#]] CrossWorkgroup

; CHECK-LABEL: Begin function call
; CHECK: %[[#P:]] = OpBitcast %[[#CharPtr]] %[[#G]]
; CHECK: %[[#C0:]] = OpCompositeInsert %[[#Q]] %[[#P]] %[[#]] 0
; CHECK: %[[#C1:]] = OpCompositeInsert %[[#Q]] %[[#]] %[[#C0]] 1
; CHECK: OpFunctionCall %[[#]] %[[#]] %[[#C1]]

; CHECK-LABEL: Begin function vector_select
; CHECK: %[[#P:]] = OpBitcast %[[#CharPtr]] %[[#G]]
; CHECK: %[[#V0:]] = OpCompositeInsert %[[#Vec]] %[[#P]] %[[#]] 0
; CHECK: %[[#V1:]] = OpCompositeInsert %[[#Vec]] %[[#P]] %[[#V0]] 1
; CHECK: OpSelect %[[#Vec]] %[[#]] %[[#]] %[[#V1]]

; CHECK-LABEL: Begin function ret
; CHECK: %[[#P:]] = OpBitcast %[[#CharPtr]] %[[#G]]
; CHECK: %[[#C0:]] = OpCompositeInsert %[[#Q]] %[[#P]] %[[#]] 0
; CHECK: %[[#C1:]] = OpCompositeInsert %[[#Q]] %[[#]] %[[#C0]] 1
; CHECK: OpReturnValue %[[#C1]]

; CHECK-LABEL: Begin function insert
; CHECK: %[[#P:]] = OpBitcast %[[#CharPtr]] %[[#G]]
; CHECK: %[[#C0:]] = OpCompositeInsert %[[#Q]] %[[#P]] %[[#]] 0
; CHECK: %[[#C1:]] = OpCompositeInsert %[[#Q]] %[[#]] %[[#C0]] 1
; CHECK: OpCompositeInsert %[[#R]] %[[#C1]] %[[#]] 0

@g = addrspace(1) global i32 0

%Q = type { ptr addrspace(1), i32 }
%R = type { %Q, i32 }

declare void @use(%Q)

define %Q @ret() {
  ret %Q { ptr addrspace(1) @g, i32 1 }
}

define void @call() {
  call void @use(%Q { ptr addrspace(1) @g, i32 1 })
  ret void
}

define %R @insert(i32 %x) {
  %r = insertvalue %R poison, %Q { ptr addrspace(1) @g, i32 1 }, 0
  %r2 = insertvalue %R %r, i32 %x, 1
  ret %R %r2
}

define <2 x ptr addrspace(1)> @vector_select(i1 %c) {
  %s = select i1 %c, <2 x ptr addrspace(1)> zeroinitializer, <2 x ptr addrspace(1)> <ptr addrspace(1) @g, ptr addrspace(1) @g>
  ret <2 x ptr addrspace(1)> %s
}

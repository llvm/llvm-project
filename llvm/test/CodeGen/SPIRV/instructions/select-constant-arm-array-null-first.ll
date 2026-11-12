; The element type of a constant array is deduced from any element: @g (i32
; pointee) after a null still makes the arm a rebuilt OpCompositeInsert chain.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpFunction %[[#Arr:]]
; CHECK: %[[#A0:]] = OpCompositeInsert %[[#Arr]] %[[#]] %[[#]] 0
; CHECK: %[[#A1:]] = OpCompositeInsert %[[#Arr]] %[[#]] %[[#A0]] 1
; CHECK: OpSelect %[[#Arr]] %[[#]] %[[#]] %[[#A1]]

@g = addrspace(1) global i32 0

define [2 x ptr addrspace(1)] @array_null_first(i1 %c) {
  %s = select i1 %c, [2 x ptr addrspace(1)] zeroinitializer, [2 x ptr addrspace(1)] [ptr addrspace(1) null, ptr addrspace(1) @g]
  ret [2 x ptr addrspace(1)] %s
}

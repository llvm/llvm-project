; A pointer in a nested struct of a constant arm: inner and outer structs are
; rebuilt with OpCompositeInsert in the result's types.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[#Q:]] = OpTypeStruct %[[#]] %[[#I32:]]
; CHECK: %[[#R:]] = OpTypeStruct %[[#Q]] %[[#I32]]

; CHECK: OpFunction %[[#R]]
; CHECK: %[[#Q0:]] = OpCompositeInsert %[[#Q]] %[[#]] %[[#]] 0
; CHECK: %[[#Q1:]] = OpCompositeInsert %[[#Q]] %[[#]] %[[#Q0]] 1
; CHECK: %[[#R0:]] = OpCompositeInsert %[[#R]] %[[#Q1]] %[[#]] 0
; CHECK: %[[#R1:]] = OpCompositeInsert %[[#R]] %[[#]] %[[#R0]] 1
; CHECK: OpSelect %[[#R]] %[[#]] %[[#]] %[[#R1]]

; CHECK: OpFunction %[[#R]]
; CHECK: %[[#PQ0:]] = OpCompositeInsert %[[#Q]] %[[#]] %[[#]] 0
; CHECK: %[[#PQ1:]] = OpCompositeInsert %[[#Q]] %[[#]] %[[#PQ0]] 1
; CHECK: %[[#PR0:]] = OpCompositeInsert %[[#R]] %[[#PQ1]] %[[#]] 0
; CHECK: %[[#PR1:]] = OpCompositeInsert %[[#R]] %[[#]] %[[#PR0]] 1
; CHECK: OpPhi %[[#R]] %[[#]] %[[#]] %[[#PR1]] %[[#]]

@g = addrspace(1) global i32 0

%Q = type { ptr addrspace(1), i32 }
%R = type { %Q, i32 }

define %R @nested(i1 %c) {
  %s = select i1 %c, %R zeroinitializer, %R { %Q { ptr addrspace(1) @g, i32 1 }, i32 2 }
  ret %R %s
}

define %R @nested_phi(i1 %c) {
entry:
  br i1 %c, label %then, label %join
then:
  br label %join
join:
  %p = phi %R [ zeroinitializer, %entry ], [ { %Q { ptr addrspace(1) @g, i32 1 }, i32 2 }, %then ]
  ret %R %p
}

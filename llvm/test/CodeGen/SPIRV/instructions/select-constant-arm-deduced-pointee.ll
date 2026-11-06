; A constant arm of an aggregate select or PHI holds a pointer whose pointee
; type is known (i32 for @g), so the constant is a different SPIR-V type than
; the result. The arm is rebuilt with insertvalue, which casts each member to
; the result's member type, so all arms have the result's type.
; A PHI arm is rebuilt in its incoming block, once even when a switch lists
; that block twice.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[#Q:]] = OpTypeStruct %[[#]] %[[#]]

; CHECK: OpFunction %[[#Q]]
; CHECK: %[[#S0:]] = OpCompositeInsert %[[#Q]] %[[#]] %[[#]] 0
; CHECK: %[[#S1:]] = OpCompositeInsert %[[#Q]] %[[#]] %[[#S0]] 1
; CHECK: OpSelect %[[#Q]] %[[#]] %[[#]] %[[#S1]]

; CHECK: OpFunction %[[#Q]]
; CHECK: %[[#P0:]] = OpCompositeInsert %[[#Q]] %[[#]] %[[#]] 0
; CHECK: %[[#P1:]] = OpCompositeInsert %[[#Q]] %[[#]] %[[#P0]] 1
; CHECK: OpPhi %[[#Q]] %[[#]] %[[#]] %[[#P1]] %[[#]]

; CHECK: OpFunction %[[#Q]]
; CHECK: OpCompositeInsert %[[#Q]] %[[#]] %[[#]] 0
; CHECK: %[[#W1:]] = OpCompositeInsert %[[#Q]] %[[#]] %[[#]] 1
; CHECK-NOT: OpCompositeInsert
; CHECK: OpPhi %[[#Q]] %[[#W1]] %[[#]] %[[#]] %[[#]]{{$}}

@g = addrspace(1) global i32 0

%Q = type { ptr addrspace(1), i32 }

define %Q @select(i1 %c) {
  %s = select i1 %c, %Q zeroinitializer, %Q { ptr addrspace(1) @g, i32 1 }
  ret %Q %s
}

define %Q @phi(i1 %c) {
entry:
  br i1 %c, label %then, label %join
then:
  br label %join
join:
  %p = phi %Q [ zeroinitializer, %entry ], [ { ptr addrspace(1) @g, i32 1 }, %then ]
  ret %Q %p
}

define %Q @switch(i32 %x) {
entry:
  switch i32 %x, label %join [
    i32 1, label %join
    i32 2, label %other
  ]
other:
  br label %join
join:
  %p = phi %Q [ { ptr addrspace(1) @g, i32 1 }, %entry ], [ { ptr addrspace(1) @g, i32 1 }, %entry ], [ zeroinitializer, %other ]
  ret %Q %p
}

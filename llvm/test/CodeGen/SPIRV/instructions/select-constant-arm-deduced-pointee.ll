; Every arm of an aggregate select has the select's type. A constant arm's
; deduced type became a distinct struct with typed pointer members once the
; inttoptr constants it holds had a pointee type deduced (here in @a, since
; constant expressions are shared between functions), which no longer matched
; the select and its zeroinitializer arm.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[#Ptr:]] = OpTypePointer Generic
; CHECK: %[[#Q:]] = OpTypeStruct %[[#Ptr]] %[[#Ptr]]
; CHECK-NOT: OpTypeStruct %[[#Ptr]] %[[#Ptr]]
; CHECK: OpSelect %[[#Q]] %[[#]] %[[#]] %[[#]]

%P = type { ptr addrspace(4), i32 }
%Q = type { ptr addrspace(4), ptr addrspace(4) }

define %P @a() {
  ret %P { ptr addrspace(4) inttoptr (i64 4 to ptr addrspace(4)), i32 0 }
}

define %Q @b(i1 %c) {
  %s = select i1 %c, %Q zeroinitializer, %Q { ptr addrspace(4) inttoptr (i64 8 to ptr addrspace(4)), ptr addrspace(4) inttoptr (i64 4 to ptr addrspace(4)) }
  ret %Q %s
}

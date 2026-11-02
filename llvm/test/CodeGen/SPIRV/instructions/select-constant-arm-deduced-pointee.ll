; A constant arm of an aggregate select or PHI holds pointer members whose
; pointee types were deduced elsewhere in the module (constant expressions are
; shared between functions): i8 for the one returned in @a, i32 for the one
; loaded through in @a2. The arm is rebuilt with insertvalue, which casts each
; member to the result's member type, so all arms have the result's type.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[#Q:]] = OpTypeStruct [[P:%[0-9]+]] [[P]]
; CHECK: OpSelect %[[#Q]] %[[#]] %[[#]] %[[#]]
; CHECK: OpSelect %[[#Q]] %[[#]] %[[#]] %[[#]]
; CHECK: OpPhi %[[#Q]]

%P = type { ptr addrspace(4), i32 }
%Q = type { ptr addrspace(4), ptr addrspace(4) }

define %P @a() {
  ret %P { ptr addrspace(4) inttoptr (i64 4 to ptr addrspace(4)), i32 0 }
}

define i32 @a2() {
  %v = load i32, ptr addrspace(4) inttoptr (i64 12 to ptr addrspace(4))
  ret i32 %v
}

define %Q @b(i1 %c) {
  %s = select i1 %c, %Q zeroinitializer, %Q { ptr addrspace(4) inttoptr (i64 8 to ptr addrspace(4)), ptr addrspace(4) inttoptr (i64 4 to ptr addrspace(4)) }
  ret %Q %s
}

define %Q @typed(i1 %c) {
  %s = select i1 %c, %Q zeroinitializer, %Q { ptr addrspace(4) inttoptr (i64 12 to ptr addrspace(4)), ptr addrspace(4) inttoptr (i64 8 to ptr addrspace(4)) }
  ret %Q %s
}

define %Q @phi(i1 %c) {
entry:
  br i1 %c, label %then, label %join
then:
  br label %join
join:
  %p = phi %Q [ zeroinitializer, %entry ], [ { ptr addrspace(4) inttoptr (i64 12 to ptr addrspace(4)), ptr addrspace(4) null }, %then ]
  ret %Q %p
}

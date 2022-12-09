; RUN: opt -S -passes=rewrite-statepoints-for-gc  < %s | FileCheck %s


; Ensure we're stipping attributes from the function signatures which are invalid
; after inserting safepoints with explicit memory semantics

declare void @f()

define i8 addrspace(1)* @deref_arg(i8 addrspace(1)* dereferenceable(16) %arg) gc "statepoint-example" {
; CHECK: define i8 addrspace(1)* @deref_arg(i8 addrspace(1)* %arg)
  call void @f()
  ret i8 addrspace(1)* %arg
}

define dereferenceable(16) i8 addrspace(1)* @deref_ret(i8 addrspace(1)* %arg) gc "statepoint-example" {
; CHECK: define i8 addrspace(1)* @deref_ret(i8 addrspace(1)* %arg)
  call void @f()
  ret i8 addrspace(1)* %arg
}

define i8 addrspace(1)* @deref_or_null_arg(i8 addrspace(1)* dereferenceable_or_null(16) %arg) gc "statepoint-example" {
; CHECK: define i8 addrspace(1)* @deref_or_null_arg(i8 addrspace(1)* %arg)
  call void @f()
  ret i8 addrspace(1)* %arg
}

define dereferenceable_or_null(16) i8 addrspace(1)* @deref_or_null_ret(i8 addrspace(1)* %arg) gc "statepoint-example" {
; CHECK: define i8 addrspace(1)* @deref_or_null_ret(i8 addrspace(1)* %arg)
  call void @f()
  ret i8 addrspace(1)* %arg
}

define i8 addrspace(1)* @noalias_arg(i8 addrspace(1)* noalias %arg) gc "statepoint-example" {
; CHECK: define i8 addrspace(1)* @noalias_arg(i8 addrspace(1)* %arg)
  call void @f()
  ret i8 addrspace(1)* %arg
}

define noalias i8 addrspace(1)* @noalias_ret(i8 addrspace(1)* %arg) gc "statepoint-example" {
; CHECK: define i8 addrspace(1)* @noalias_ret(i8 addrspace(1)* %arg)
  call void @f()
  ret i8 addrspace(1)* %arg
}

define i8 addrspace(1)* @nofree(i8 addrspace(1)* nofree %arg) nofree gc "statepoint-example" {
; CHECK: define i8 addrspace(1)* @nofree(i8 addrspace(1)* %arg) gc "statepoint-example" {
  call void @f()
  ret i8 addrspace(1)* %arg
}

define i8 addrspace(1)* @nosync(i8 addrspace(1)* %arg) nosync gc "statepoint-example" {
; CHECK: define i8 addrspace(1)* @nosync(i8 addrspace(1)* %arg) gc "statepoint-example" {
  call void @f()
  ret i8 addrspace(1)* %arg
}

define i8 addrspace(1)* @readnone(i8 addrspace(1)* readnone %arg) readnone gc "statepoint-example" {
; CHECK: define i8 addrspace(1)* @readnone(i8 addrspace(1)* %arg) gc "statepoint-example" {
  call void @f()
  ret i8 addrspace(1)* %arg
}

define i8 addrspace(1)* @readonly(i8 addrspace(1)* readonly %arg) readonly gc "statepoint-example" {
; CHECK: define i8 addrspace(1)* @readonly(i8 addrspace(1)* %arg) gc "statepoint-example" {
  call void @f()
  ret i8 addrspace(1)* %arg
}

define i8 addrspace(1)* @writeonly(i8 addrspace(1)* writeonly %arg) writeonly gc "statepoint-example" {
; CHECK: define i8 addrspace(1)* @writeonly(i8 addrspace(1)* %arg) gc "statepoint-example" {
  call void @f()
  ret i8 addrspace(1)* %arg
}

define i8 addrspace(1)* @argmemonly(i8 addrspace(1)* %arg) argmemonly gc "statepoint-example" {
; CHECK: define i8 addrspace(1)* @argmemonly(i8 addrspace(1)* %arg) gc "statepoint-example" {
  call void @f()
  ret i8 addrspace(1)* %arg
}

define i8 addrspace(1)* @inaccessiblememonly(i8 addrspace(1)* %arg) inaccessiblememonly gc "statepoint-example" {
; CHECK: define i8 addrspace(1)* @inaccessiblememonly(i8 addrspace(1)* %arg) gc "statepoint-example" {
  call void @f()
  ret i8 addrspace(1)* %arg
}

define i8 addrspace(1)* @inaccessiblemem_or_argmemonly(i8 addrspace(1)* %arg) inaccessiblemem_or_argmemonly gc "statepoint-example" {
; CHECK: define i8 addrspace(1)* @inaccessiblemem_or_argmemonly(i8 addrspace(1)* %arg) gc "statepoint-example" {
  call void @f()
  ret i8 addrspace(1)* %arg
}


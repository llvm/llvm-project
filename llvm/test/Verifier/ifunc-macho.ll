; RUN:  not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

target triple = "arm64-apple-ios"

define ptr @resolver() {
  ret ptr null
}

@g = external global i32
@inval_objtype = ifunc void (), ptr @g
; CHECK: IFunc must have a Function resolver

declare ptr @resolver_decl()
@inval_resolver_decl = ifunc void (), ptr @resolver_decl
; CHECK: IFunc resolver must be a definition
; CHECK-NEXT: @inval_resolver_decl

define available_externally ptr @resolver_linker_decl() {
  ret ptr null
}
@inval_resolver_decl2 = ifunc void (), ptr @resolver_linker_decl
; CHECK: IFunc resolver must be a definition
; CHECK-NEXT: @inval_resolver_decl2

@ifunc_nonpointer_return_type = ifunc i32 (), ptr @resolver_returns_nonpointer
; CHECK: IFunc resolver must return a pointer
; CHECK-NEXT: ptr @ifunc_nonpointer_return_type

define i32 @resolver_returns_nonpointer() {
  ret i32 0
}

@valid_external = ifunc void (), ptr @resolver
; CHECK-NOT: valid_external

@inval_linkonce = linkonce ifunc void (), ptr @resolver

@inval_weak = weak ifunc void (), ptr @resolver

@inval_weak_extern = extern_weak ifunc void (), ptr @resolver

@inval_private = private ifunc void (), ptr @resolver

; RUN:  not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

define void ()* @resolver() {
  ret void ()* null
}

@inval_linkage = extern_weak ifunc void (), void ()* ()* @resolver
; CHECK: IFunc should have {{.*}} linkage!
; CHECK-NEXT: @inval_linkage

@g = external global i32
@inval_objtype = ifunc void (), bitcast(i32* @g to void ()* ()*)
; CHECK: IFunc must have a Function resolver

declare void ()* @resolver_decl()
@inval_resolver_decl = ifunc void (), void ()* ()* @resolver_decl
; CHECK: IFunc resolver must be a definition
; CHECK-NEXT: @inval_resolver_decl

define available_externally void ()* @resolver_linker_decl() {
  ret void ()* null
}
@inval_resolver_decl2 = ifunc void (), void ()* ()* @resolver_linker_decl
; CHECK: IFunc resolver must be a definition
; CHECK-NEXT: @inval_resolver_decl2

@inval_resolver_type = ifunc i32 (), void ()* ()* @resolver
; CHECK: IFunc resolver has incorrect type
; CHECK-NEXT: @inval_resolver_type

@ifunc_nonpointer_return_type = ifunc i32 (), i32 ()* @resolver_returns_nonpointer
; CHECK: IFunc resolver must return a pointer
; CHECK-NEXT: i32 ()* @ifunc_nonpointer_return_type

define i32 @resolver_returns_nonpointer() {
  ret i32 0
}

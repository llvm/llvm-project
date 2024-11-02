; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; Check constantexprs which ifunc looks through to find the resolver
; function.

@ifunc_addrspacecast_as1_to_as0 = ifunc void (), ptr addrspacecast (ptr addrspace(1) @resolver_as1 to ptr)

; CHECK: @alias_resolver = internal alias i32 (i32), ptr @resolver
@alias_resolver = internal alias i32 (i32), ptr @resolver

; CHECK: @ifunc_resolver_is_alias = internal ifunc i32 (i32), ptr @alias_resolver
@ifunc_resolver_is_alias = internal ifunc i32 (i32), ptr @alias_resolver


; CHECK: define ptr @resolver_as1() addrspace(1) {
define ptr @resolver_as1() addrspace(1) {
  ret ptr null
}

; CHECK: define internal ptr @resolver() {
define internal ptr @resolver() {
  ret ptr null
}

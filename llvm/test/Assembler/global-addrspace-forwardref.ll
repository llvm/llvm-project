; RUN: llvm-as --opaque-pointers=0 < %s | llvm-dis --opaque-pointers=0 | FileCheck %s
; RUN: verify-uselistorder --opaque-pointers=0 %s

; Make sure the address space of forward decls is preserved

; CHECK: @a2 = global i8 addrspace(1)* @a
; CHECK: @a = addrspace(1) global i8 0
@a2 = global i8 addrspace(1)* @a
@a = addrspace(1) global i8 0

; Now test with global IDs instead of global names.

; CHECK: @a3 = global i8 addrspace(1)* @0
; CHECK: @0 = addrspace(1) global i8 0

@a3 = global i8 addrspace(1)* @0
@0 = addrspace(1) global i8 0


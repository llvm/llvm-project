; RUN: llvm-as  < %s | llvm-dis  | FileCheck %s
; RUN: verify-uselistorder  %s

; Make sure the address space of forward decls is preserved

; CHECK: @a2 = global ptr addrspace(1) @a
; CHECK: @a = addrspace(1) global i8 0
@a2 = global ptr addrspace(1) @a
@a = addrspace(1) global i8 0

; Now test with global IDs instead of global names.

; CHECK: @a3 = global ptr addrspace(1) @0
; CHECK: @0 = addrspace(1) global i8 0

@a3 = global ptr addrspace(1) @0
@0 = addrspace(1) global i8 0


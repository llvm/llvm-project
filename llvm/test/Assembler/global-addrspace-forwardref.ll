; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; Make sure the address space of forward decls is preserved

; CHECK: @a2 = global i8 addrspace(1)* @a
; CHECK: @a = addrspace(1) global i8 0
@a2 = global i8 addrspace(1)* @a
@a = addrspace(1) global i8 0

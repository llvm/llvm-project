; RUN: llc < %s --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types | FileCheck %s

%externref = type ptr addrspace(10) ;; addrspace 10 is nonintegral
%funcref = type ptr addrspace(20)   ;; addrspace 20 is nonintegral

; CHECK: .tabletype eref_table, externref
; CHECK-NEXT: .globl eref_table
; CHECK-LABEL: eref_table:
@eref_table = local_unnamed_addr addrspace(1) global [0 x %externref] undef

; CHECK-NOT: .globl .Lprivate_eref_table
; CHECK: .tabletype .Lprivate_eref_table, externref
; CHECK-LABEL: .Lprivate_eref_table:
@private_eref_table = private local_unnamed_addr addrspace(1) global [0 x %externref] undef

; CHECK: .tabletype extern_eref_table, externref
; CHECK-NOT: .globl extern_eref_table
; CHECK-NOT: extern_eref_table:
@extern_eref_table = external addrspace(1) global [0 x %externref]

; CHECK: .tabletype fref_table, funcref
; CHECK-NEXT: .globl fref_table
; CHECK-LABEL: fref_table:
@fref_table = local_unnamed_addr addrspace(1) global [0 x %funcref] undef

; CHECK-NOT: .globl .Lprivate_fref_table
; CHECK: .tabletype .Lprivate_fref_table, funcref
; CHECK-LABEL: .Lprivate_fref_table:
@private_fref_table = private local_unnamed_addr addrspace(1) global [0 x %funcref] undef

; CHECK: .tabletype extern_fref_table, funcref
; CHECK-NOT: .globl extern_fref_table
; CHECK-NOT: extern_fref_table:
@extern_fref_table = external addrspace(1) global [0 x %funcref]

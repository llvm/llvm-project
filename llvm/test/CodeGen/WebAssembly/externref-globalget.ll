; RUN: llc < %s --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types | FileCheck %s

%externref = type ptr addrspace(10) ;; addrspace 10 is nonintegral

@externref_global = local_unnamed_addr addrspace(1) global %externref undef

define %externref @return_externref_global() {
  ;; this generates a global.get of @externref_global
  %ref = load %externref, ptr addrspace(1) @externref_global
  ret %externref %ref
}

; CHECK-LABEL: return_externref_global:
; CHECK-NEXT: functype       return_externref_global () -> (externref)
; CHECK-NEXT: global.get externref_global
; CHECK-NEXT: end_function

; CHECK: .globl externref_global

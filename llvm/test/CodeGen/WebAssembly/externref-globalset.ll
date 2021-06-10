; RUN: llc --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types < %s | FileCheck %s

%extern = type opaque
%externref = type %extern addrspace(10)* ;; addrspace 10 is nonintegral

@externref_global = local_unnamed_addr addrspace(1) global %externref undef

define void @set_externref_global(%externref %g) {
  ;; this generates a global.set of @externref.global
  store %externref %g, %externref addrspace(1)* @externref_global
  ret void
}

; CHECK-LABEL: set_externref_global:
; CHECK-NEXT: functype       set_externref_global (externref) -> ()
; CHECK-NEXT: local.get      0
; CHECK-NEXT: global.set     externref_global
; CHECK-NEXT: end_function

; CHECK: .globl externref_global

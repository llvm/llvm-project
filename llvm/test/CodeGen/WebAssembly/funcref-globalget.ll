; RUN: llc < %s --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types | FileCheck %s

%funcref = type i8 addrspace(20)* ;; addrspace 20 is nonintegral

@funcref_global = local_unnamed_addr addrspace(1) global %funcref undef

define %funcref @return_funcref_global() {
  ;; this generates a global.get of @funcref_global
  %ref = load %funcref, %funcref addrspace(1)* @funcref_global
  ret %funcref %ref
}

; CHECK-LABEL: return_funcref_global:
; CHECK-NEXT: .functype       return_funcref_global () -> (funcref)
; CHECK-NEXT: global.get funcref_global
; CHECK-NEXT: end_function

; CHECK: .globl funcref_global

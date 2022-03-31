; RUN: llc < %s --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types | FileCheck %s

%funcref = type i8 addrspace(20)* ;; addrspace 20 is nonintegral

@funcref_global = local_unnamed_addr addrspace(1) global %funcref undef

define void @set_funcref_global(%funcref %g) {
  ;; this generates a global.set of @funcref_global
  store %funcref %g, %funcref addrspace(1)* @funcref_global
  ret void
}

; CHECK-LABEL: set_funcref_global:
; CHECK-NEXT: functype       set_funcref_global (funcref) -> ()
; CHECK-NEXT: local.get      0
; CHECK-NEXT: global.set     funcref_global
; CHECK-NEXT: end_function

; CHECK: .globl funcref_global

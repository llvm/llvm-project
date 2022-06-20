; RUN: llc < %s --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types 2>&1 | FileCheck %s

%externref = type ptr addrspace(10)

define %externref @int_to_externref(i32 %i) {
  %ref = inttoptr i32 %i to %externref
  ret %externref %ref
}


; CHECK-LABEL: int_to_externref:
; CHECK-NEXT: .functype       int_to_externref (i32) -> (externref)
; CHECK-NEXT: .local externref
; CHECK-NEXT: unreachable
; CHECK-NEXT: local.get 1
; CHECK-NEXT: end_function

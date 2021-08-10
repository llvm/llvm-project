; RUN: llc < %s --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types 2>&1 | FileCheck %s

%extern = type opaque
%externref = type %extern addrspace(10)*

define i32 @externref_to_int(%externref %ref) {
  %i = ptrtoint %externref %ref to i32
  ret i32 %i
}

; CHECK-LABEL: externref_to_int:
; CHECK-NEXT: .functype       externref_to_int (externref) -> (i32)
; CHECK-NEXT: .local i32
; CHECK-NEXT: unreachable
; CHECK-NEXT: local.get 1
; CHECK-NEXT: end_function

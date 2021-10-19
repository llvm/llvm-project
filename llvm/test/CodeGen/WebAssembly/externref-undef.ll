; RUN: llc < %s --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types | FileCheck %s

%extern = type opaque
%externref = type %extern addrspace(10)* ;; addrspace 10 is nonintegral

@externref_global = local_unnamed_addr addrspace(1) global %externref undef

define %extern @return_extern_undef() {
  ; Returning a ref.null or an uninitialized externref would make
  ; more sense if the return type would be %externref. However, in
  ; this case this is an %extern value, which really is an opaque
  ; type and should never really happen.
  ret %extern undef
}

; CHECK-LABEL: return_extern_undef:
; CHECK-NEXT: functype       return_extern_undef () -> ()
; CHECK-NEXT: end_function

; CHECK: .globl externref_global


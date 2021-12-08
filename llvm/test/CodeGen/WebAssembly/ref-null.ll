; RUN: llc --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types < %s | FileCheck %s

%extern = type opaque
%externref = type %extern addrspace(10)* ;; addrspace 10 is nonintegral
%funcref = type i8 addrspace(20)*      ;; addrspace 20 is nonintegral

declare %externref @llvm.wasm.ref.null.extern() nounwind
declare %funcref @llvm.wasm.ref.null.func() nounwind

define %externref @get_null_extern() {
; CHECK-LABEL: get_null_extern:
; CHECK-NEXT:  .functype       get_null_extern () -> (externref)
; CHECK-NEXT:  ref.null_extern
; CHECK-NEXT:  end_function
  %null = call %externref @llvm.wasm.ref.null.extern()
  ret %externref %null
}

define %funcref @get_null_func() {
; CHECK-LABEL: get_null_func:
; CHECK-NEXT:  .functype       get_null_func () -> (funcref)
; CHECK-NEXT:  ref.null_func
; CHECK-NEXT:  end_function
  %null = call %funcref @llvm.wasm.ref.null.func()
  ret %funcref %null
}

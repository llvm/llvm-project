; RUN: not llc --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types < %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

%externref = type ptr addrspace(10)

define void @store_extern(%externref %ref) {
  store %extern undef, %externref %ref
  ret void
}

; CHECK-ERROR: error: storing unsized types is not allowed

; RUN: not --crash llc --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types < %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

%extern = type opaque
%externref = type %extern addrspace(10)*

define %externref @int_to_externref(i32 %i) {
  %ref = inttoptr i32 %i to %externref
  ret %externref %ref
}

; CHECK-ERROR: LLVM ERROR: inttoptr not allowed on reference types

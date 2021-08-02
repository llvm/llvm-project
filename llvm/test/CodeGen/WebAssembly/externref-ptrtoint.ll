; RUN: not --crash llc --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types < %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

%extern = type opaque
%externref = type %extern addrspace(10)*

define i32 @externref_to_int(%externref %ref) {
  %i = ptrtoint %externref %ref to i32
  ret i32 %i
}

; CHECK-ERROR: LLVM ERROR: ptrtoint not allowed on reference types

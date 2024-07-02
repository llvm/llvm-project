; RUN: not llc --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types < %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

%externref = type target("wasm.externref")

define %externref @int_to_externref(i32 %i) {
  %ref = inttoptr i32 %i to %externref
  ret %externref %ref
}

# CHECK-ERROR: error: invalid cast opcode for cast from 'i32' to 'target("wasm.externref")'
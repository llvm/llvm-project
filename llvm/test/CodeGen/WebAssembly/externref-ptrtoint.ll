; RUN: not llc --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types < %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

%externref = type target("wasm.externref")

define i32 @externref_to_int(%externref %ref) {
  %i = ptrtoint %externref %ref to i32
  ret i32 %i
}

# CHECK-ERROR: error: invalid cast opcode for cast from 'target("wasm.externref")' to 'i32'

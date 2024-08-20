; RUN: not llc --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types < %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

%externref = type target("wasm.funcref")

define void @store_extern(%externref %ref) {
  store %extern undef, %externref %ref
  ret void
}

; CHECK-ERROR: error: store operand must be a pointer

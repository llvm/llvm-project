; Check that we accept functions with '$' in the name.

; RUN: llc -mtriple=wasm32-unknown-linux < %s | FileCheck %s
; RUN: llc -mtriple=wasm32-unknown-linux -asm-verbose=false < %s | FileCheck %s --check-prefix=CHECK-NOVERBOSE

define hidden i32 @"_Z54bar$ompvariant$bar"() {
entry:
  ret i32 2
}

; RUN: llc --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types < %s 2>&1 | FileCheck %s

%externref = type ptr addrspace(10)

; CHECK-LABEL: load_ref:
; CHECK-NEXT: .functype load_ref (i32) -> (externref)
; CHECK-NEXT: local.get 0
; CHECK-NEXT: table.get	__externref_table
define %externref @load_ref(ptr %p) {
entry:
  %1 = load %externref, ptr %p
  ret %externref %1
}

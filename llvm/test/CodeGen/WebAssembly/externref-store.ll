; RUN: llc --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types < %s 2>&1 | FileCheck %s

%externref = type ptr addrspace(10)

; CHECK-LABEL: store_ref:
; CHECK-NEXT: .functype store_ref (i32, externref) -> ()
; CHECK-NEXT: local.get 0
; CHECK-NEXT: local.get 1
; CHECK-NEXT: table.set	__externref_table
define void @store_ref(ptr %p, %externref %x) {
entry:
  store %externref %x, ptr %p
  ret void
}

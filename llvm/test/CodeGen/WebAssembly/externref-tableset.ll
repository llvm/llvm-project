; RUN: llc --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types < %s | FileCheck %s

%externref = type ptr addrspace(10) ;; addrspace 10 is nonintegral

@externref_table = local_unnamed_addr addrspace(1) global [0 x %externref] undef

declare void @llvm.wasm.table.set.externref(ptr addrspace(1), i32, %externref) nounwind

define void @set_externref_table(%externref %g, i32 %i) {
; CHECK-LABEL: set_externref_table:
; CHECK-NEXT: .functype       set_externref_table (externref, i32) -> ()
; CHECK-NEXT: local.get      1
; CHECK-NEXT: local.get      0
; CHECK-NEXT: table.set     externref_table
; CHECK-NEXT: end_function

;; this generates a table.set of @externref_table
  call void @llvm.wasm.table.set.externref(ptr addrspace(1) @externref_table, i32 %i, %externref %g)
  ret void
}

define void @set_externref_table_const(%externref %g) {
; CHECK-LABEL: set_externref_table_const:
; CHECK-NEXT:  .functype      set_externref_table_const (externref) -> ()
; CHECK-NEXT:  i32.const      0
; CHECK-NEXT:  local.get      0
; CHECK-NEXT:  table.set      externref_table
; CHECK-NEXT:  end_function
  call void @llvm.wasm.table.set.externref(ptr addrspace(1) @externref_table, i32 0, %externref %g)
  ret void
}

define void @set_externref_table_with_offset(%externref %g, i32 %i) {
; CHECK-LABEL: set_externref_table_with_offset:
; CHECK-NEXT:  .functype       set_externref_table_with_offset (externref, i32) -> ()
; CHECK-NEXT:  local.get       1
; CHECK-NEXT:  i32.const       2
; CHECK-NEXT:  i32.add
; CHECK-NEXT:  local.get       0
; CHECK-NEXT:  table.set       externref_table
; CHECK-NEXT:  end_function
  %off = add nsw i32 %i, 2
  call void @llvm.wasm.table.set.externref(ptr addrspace(1) @externref_table, i32 %off, %externref %g)
  ret void
}

define void @set_externref_table_with_var_offset(%externref %g, i32 %i, i32 %j) {
; CHECK-LABEL: set_externref_table_with_var_offset:
; CHECK-NEXT:  .functype       set_externref_table_with_var_offset (externref, i32, i32) -> ()
; CHECK-NEXT:  local.get       1
; CHECK-NEXT:  local.get       2
; CHECK-NEXT:  i32.add
; CHECK-NEXT:  local.get       0
; CHECK-NEXT:  table.set       externref_table
; CHECK-NEXT:  end_function
  %off = add nsw i32 %i, %j
  call void @llvm.wasm.table.set.externref(ptr addrspace(1) @externref_table, i32 %off, %externref %g)
  ret void
}

declare i32 @set_offset()

define void @set_externref_table_with_var_offset2(%externref %g, i32 %i) {
; CHECK-LABEL: set_externref_table_with_var_offset2:
; CHECK-NEXT:  .functype       set_externref_table_with_var_offset2 (externref, i32) -> ()
; CHECK-NEXT:  local.get       1
; CHECK-NEXT:  call    set_offset
; CHECK-NEXT:  i32.add
; CHECK-NEXT:  local.get       0
; CHECK-NEXT:  table.set       externref_table
; CHECK-NEXT:  end_function
  %j = call i32 @set_offset()
  %off = add nsw i32 %i, %j
  call void @llvm.wasm.table.set.externref(ptr addrspace(1) @externref_table, i32 %off, %externref %g)
  ret void
}

declare i32 @get_table_slot() local_unnamed_addr

define void @set_externref_table_with_id_from_call(%externref %g) {
; CHECK-LABEL: set_externref_table_with_id_from_call:
; CHECK-NEXT:  .functype       set_externref_table_with_id_from_call (externref) -> ()
; CHECK-NEXT:  call    get_table_slot
; CHECK-NEXT:  local.get       0
; CHECK-NEXT:  table.set       externref_table
; CHECK-NEXT:  end_function
  %id = call i32 @get_table_slot()
  call void @llvm.wasm.table.set.externref(ptr addrspace(1) @externref_table, i32 %id, %externref %g)
  ret void
}

;       CHECK: .tabletype externref_table, externref
; CHECK-LABEL: externref_table:

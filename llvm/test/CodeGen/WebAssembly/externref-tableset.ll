; RUN: sed 's/iX/i32/g' < %s | llc --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types | FileCheck %s  -DiPTR=i32
; RUN: sed 's/iX/i64/g' < %s | llc --mtriple=wasm64-unknown-unknown -asm-verbose=false -mattr=+reference-types | FileCheck %s  -DiPTR=i64


%externref = type ptr addrspace(10) ;; addrspace 10 is nonintegral

@externref_table = local_unnamed_addr addrspace(1) global [0 x %externref] undef

declare void @llvm.wasm.table.set.externref(ptr addrspace(1), iX, %externref) nounwind

define void @set_externref_table(%externref %g, iX %i) {
; CHECK-LABEL: set_externref_table:
; CHECK-NEXT:  .functype       set_externref_table (externref, [[iPTR]]) -> ()
; CHECK-NEXT:  local.get      1
; CHECK-NEXT:  local.get      0
; CHECK-NEXT:  table.set     externref_table
; CHECK-NEXT:  end_function

;; this generates a table.set of @externref_table
  call void @llvm.wasm.table.set.externref(ptr addrspace(1) @externref_table, iX %i, %externref %g)
  ret void
}

define void @set_externref_table_const(%externref %g) {
; CHECK-LABEL: set_externref_table_const:
; CHECK-NEXT:  .functype      set_externref_table_const (externref) -> ()
; CHECK-NEXT:  [[iPTR]].const      0
; CHECK-NEXT:  local.get      0
; CHECK-NEXT:  table.set      externref_table
; CHECK-NEXT:  end_function
  call void @llvm.wasm.table.set.externref(ptr addrspace(1) @externref_table, iX 0, %externref %g)
  ret void
}

define void @set_externref_table_with_offset(%externref %g, iX %i) {
; CHECK-LABEL: set_externref_table_with_offset:
; CHECK-NEXT:  .functype       set_externref_table_with_offset (externref, [[iPTR]]) -> ()
; CHECK-NEXT:  local.get       1
; CHECK-NEXT:  [[iPTR]].const  2
; CHECK-NEXT:  [[iPTR]].add
; CHECK-NEXT:  local.get       0
; CHECK-NEXT:  table.set       externref_table
; CHECK-NEXT:  end_function
  %off = add nsw iX %i, 2
  call void @llvm.wasm.table.set.externref(ptr addrspace(1) @externref_table, iX %off, %externref %g)
  ret void
}

define void @set_externref_table_with_var_offset(%externref %g, iX %i, iX %j) {
; CHECK-LABEL: set_externref_table_with_var_offset:
; CHECK-NEXT:  .functype       set_externref_table_with_var_offset (externref, [[iPTR]], [[iPTR]]) -> ()
; CHECK-NEXT:  local.get       1
; CHECK-NEXT:  local.get       2
; CHECK-NEXT:  [[iPTR]].add
; CHECK-NEXT:  local.get       0
; CHECK-NEXT:  table.set       externref_table
; CHECK-NEXT:  end_function
  %off = add nsw iX %i, %j
  call void @llvm.wasm.table.set.externref(ptr addrspace(1) @externref_table, iX %off, %externref %g)
  ret void
}

declare iX @set_offset()

define void @set_externref_table_with_var_offset2(%externref %g, iX %i) {
; CHECK-LABEL: set_externref_table_with_var_offset2:
; CHECK-NEXT:  .functype       set_externref_table_with_var_offset2 (externref, [[iPTR]]) -> ()
; CHECK-NEXT:  local.get       1
; CHECK-NEXT:  call    set_offset
; CHECK-NEXT:  [[iPTR]].add
; CHECK-NEXT:  local.get       0
; CHECK-NEXT:  table.set       externref_table
; CHECK-NEXT:  end_function
  %j = call iX @set_offset()
  %off = add nsw iX %i, %j
  call void @llvm.wasm.table.set.externref(ptr addrspace(1) @externref_table, iX %off, %externref %g)
  ret void
}

declare iX @get_table_slot() local_unnamed_addr

define void @set_externref_table_with_id_from_call(%externref %g) {
; CHECK-LABEL: set_externref_table_with_id_from_call:
; CHECK-NEXT:  .functype       set_externref_table_with_id_from_call (externref) -> ()
; CHECK-NEXT:  call    get_table_slot
; CHECK-NEXT:  local.get       0
; CHECK-NEXT:  table.set       externref_table
; CHECK-NEXT:  end_function
  %id = call iX @get_table_slot()
  call void @llvm.wasm.table.set.externref(ptr addrspace(1) @externref_table, iX %id, %externref %g)
  ret void
}

;       CHECK: .tabletype externref_table, externref
; CHECK-LABEL: externref_table:

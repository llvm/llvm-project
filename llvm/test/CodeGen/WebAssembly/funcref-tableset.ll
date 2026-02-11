; RUN: sed 's/iX/i32/g' < %s | llc --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types | FileCheck %s -DiPTR=i32
; RUN: sed 's/iX/i64/g' < %s | llc --mtriple=wasm64-unknown-unknown -asm-verbose=false -mattr=+reference-types | FileCheck %s -DiPTR=i64

%funcref = type ptr addrspace(20) ;; addrspace 20 is nonintegral

@funcref_table = local_unnamed_addr addrspace(1) global [0 x %funcref] undef

declare void @llvm.wasm.table.set.funcref(ptr addrspace(1), iX, %funcref) nounwind

define void @set_funcref_table(%funcref %g, iX %i) {
; CHECK-LABEL: set_funcref_table:
; CHECK-NEXT:  .functype       set_funcref_table (funcref, [[iPTR]]) -> ()
; CHECK-NEXT:  local.get      1
; CHECK-NEXT:  local.get      0
; CHECK-NEXT:  table.set     funcref_table
; CHECK-NEXT:  end_function

;; this generates a table.set of @funcref_table
  call void @llvm.wasm.table.set.funcref(ptr addrspace(1) @funcref_table, iX %i, %funcref %g)
  ret void
}

define void @set_funcref_table_const(%funcref %g) {
; CHECK-LABEL: set_funcref_table_const:
; CHECK-NEXT:  .functype      set_funcref_table_const (funcref) -> ()
; CHECK-NEXT:  [[iPTR]].const 0
; CHECK-NEXT:  local.get      0
; CHECK-NEXT:  table.set      funcref_table
; CHECK-NEXT:  end_function
  call void @llvm.wasm.table.set.funcref(ptr addrspace(1) @funcref_table, iX 0, %funcref %g)
  ret void
}

define void @set_funcref_table_with_offset(%funcref %g, iX %i) {
; CHECK-LABEL: set_funcref_table_with_offset:
; CHECK-NEXT:  .functype       set_funcref_table_with_offset (funcref, [[iPTR]]) -> ()
; CHECK-NEXT:  local.get       1
; CHECK-NEXT:  [[iPTR]].const  2
; CHECK-NEXT:  [[iPTR]].add
; CHECK-NEXT:  local.get       0
; CHECK-NEXT:  table.set       funcref_table
; CHECK-NEXT:  end_function
  %off = add nsw iX %i, 2
  call void @llvm.wasm.table.set.funcref(ptr addrspace(1) @funcref_table, iX %off, %funcref %g)
  ret void
}

define void @set_funcref_table_with_var_offset(%funcref %g, iX %i, iX %j) {
; CHECK-LABEL: set_funcref_table_with_var_offset:
; CHECK-NEXT:  .functype       set_funcref_table_with_var_offset (funcref, [[iPTR]], [[iPTR]]) -> ()
; CHECK-NEXT:  local.get       1
; CHECK-NEXT:  local.get       2
; CHECK-NEXT:  [[iPTR]].add
; CHECK-NEXT:  local.get       0
; CHECK-NEXT:  table.set       funcref_table
; CHECK-NEXT:  end_function
  %off = add nsw iX %i, %j
  call void @llvm.wasm.table.set.funcref(ptr addrspace(1) @funcref_table, iX %off, %funcref %g)
  ret void
}

declare iX @set_offset()

define void @set_funcref_table_with_var_offset2(%funcref %g, iX %i) {
; CHECK-LABEL: set_funcref_table_with_var_offset2:
; CHECK-NEXT:  .functype       set_funcref_table_with_var_offset2 (funcref, [[iPTR]]) -> ()
; CHECK-NEXT:  local.get       1
; CHECK-NEXT:  call    set_offset
; CHECK-NEXT:  [[iPTR]].add
; CHECK-NEXT:  local.get       0
; CHECK-NEXT:  table.set       funcref_table
; CHECK-NEXT:  end_function
  %j = call iX @set_offset()
  %off = add nsw iX %i, %j
  call void @llvm.wasm.table.set.funcref(ptr addrspace(1) @funcref_table, iX %off, %funcref %g)
  ret void
}

declare iX @get_table_slot() local_unnamed_addr

define void @set_funcref_table_with_id_from_call(%funcref %g) {
; CHECK-LABEL: set_funcref_table_with_id_from_call:
; CHECK-NEXT:  .functype       set_funcref_table_with_id_from_call (funcref) -> ()
; CHECK-NEXT:  call    get_table_slot
; CHECK-NEXT:  local.get       0
; CHECK-NEXT:  table.set       funcref_table
; CHECK-NEXT:  end_function
  %id = call iX @get_table_slot()
  call void @llvm.wasm.table.set.funcref(ptr addrspace(1) @funcref_table, iX %id, %funcref %g)
  ret void
}

;       CHECK: .tabletype funcref_table, funcref
; CHECK-LABEL: funcref_table:

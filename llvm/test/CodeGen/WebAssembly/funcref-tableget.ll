; RUN: llc < %s --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types | FileCheck %s

%funcref = type ptr addrspace(20) ;; addrspace 20 is nonintegral

@funcref_table = local_unnamed_addr addrspace(1) global [0 x %funcref] undef

declare %funcref @llvm.wasm.table.get.funcref(ptr addrspace(1), i32) nounwind

define %funcref @get_funcref_from_table(i32 %i) {
; CHECK-LABEL: get_funcref_from_table:
; CHECK-NEXT: .functype       get_funcref_from_table (i32) -> (funcref)
; CHECK-NEXT: local.get 0
; CHECK-NEXT: table.get funcref_table
; CHECK-NEXT: end_function
  %ref = call %funcref @llvm.wasm.table.get.funcref(ptr addrspace(1) @funcref_table, i32 %i)
  ret %funcref %ref
}

define %funcref @get_funcref_from_table_const() {
; CHECK-LABEL: get_funcref_from_table_const:
; CHECK-NEXT:  .functype      get_funcref_from_table_const () -> (funcref)
; CHECK-NEXT:  i32.const      0
; CHECK-NEXT:  table.get      funcref_table
; CHECK-NEXT:  end_function
  %ref = call %funcref @llvm.wasm.table.get.funcref(ptr addrspace(1) @funcref_table, i32 0)
  ret %funcref %ref
}

define %funcref @get_funcref_from_table_with_offset(i32 %i) {
; CHECK-LABEL: get_funcref_from_table_with_offset:
; CHECK-NEXT:  .functype       get_funcref_from_table_with_offset (i32) -> (funcref)
; CHECK-NEXT:  local.get       0
; CHECK-NEXT:  i32.const       2
; CHECK-NEXT:  i32.add
; CHECK-NEXT:  table.get       funcref_table
; CHECK-NEXT:  end_function
  %off = add nsw i32 %i, 2
  %ref = call %funcref @llvm.wasm.table.get.funcref(ptr addrspace(1) @funcref_table, i32 %off)
  ret %funcref %ref
}


define %funcref @get_funcref_from_table_with_var_offset(i32 %i, i32 %j) {
; CHECK-LABEL: get_funcref_from_table_with_var_offset:
; CHECK-NEXT:  .functype       get_funcref_from_table_with_var_offset (i32, i32) -> (funcref)
; CHECK-NEXT:  local.get       0
; CHECK-NEXT:  local.get       1
; CHECK-NEXT:  i32.add
; CHECK-NEXT:  table.get       funcref_table
; CHECK-NEXT:  end_function
  %off = add nsw i32 %i, %j
  %ref = call %funcref @llvm.wasm.table.get.funcref(ptr addrspace(1) @funcref_table, i32 %off)
  ret %funcref %ref
}

declare i32 @get_offset()

define %funcref @get_funcref_from_table_with_var_offset2(i32 %i) {
; CHECK-LABEL: get_funcref_from_table_with_var_offset2:
; CHECK-NEXT:  .functype       get_funcref_from_table_with_var_offset2 (i32) -> (funcref)
; CHECK-NEXT:  local.get       0
; CHECK-NEXT:  call    get_offset
; CHECK-NEXT:  i32.add
; CHECK-NEXT:  table.get       funcref_table
; CHECK-NEXT:  end_function
  %j = call i32 @get_offset()
  %off = add nsw i32 %i, %j
  %ref = call %funcref @llvm.wasm.table.get.funcref(ptr addrspace(1) @funcref_table, i32 %off)
  ret %funcref %ref
}

;       CHECK: .tabletype funcref_table, funcref
; CHECK-LABEL: funcref_table:

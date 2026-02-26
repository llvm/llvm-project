; RUN: sed 's/iX/i32/g' < %s | llc --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types | FileCheck %s -DiPTR=i32
; RUN: sed 's/iX/i64/g' < %s | llc --mtriple=wasm64-unknown-unknown -asm-verbose=false -mattr=+reference-types | FileCheck %s -DiPTR=i64

%funcref = type ptr addrspace(20) ;; addrspace 20 is nonintegral

@funcref_table = local_unnamed_addr addrspace(1) global [0 x %funcref] undef

declare %funcref @llvm.wasm.table.get.funcref(ptr addrspace(1), iX) nounwind

define %funcref @get_funcref_from_table(iX %i) {
; CHECK-LABEL: get_funcref_from_table:
; CHECK-NEXT:  .functype       get_funcref_from_table ([[iPTR]]) -> (funcref)
; CHECK-NEXT:  local.get 0
; CHECK-NEXT:  table.get funcref_table
; CHECK-NEXT:  end_function
  %ref = call %funcref @llvm.wasm.table.get.funcref(ptr addrspace(1) @funcref_table, iX %i)
  ret %funcref %ref
}

define %funcref @get_funcref_from_table_const() {
; CHECK-LABEL: get_funcref_from_table_const:
; CHECK-NEXT:  .functype      get_funcref_from_table_const () -> (funcref)
; CHECK-NEXT:  [[iPTR]].const 0
; CHECK-NEXT:  table.get      funcref_table
; CHECK-NEXT:  end_function
  %ref = call %funcref @llvm.wasm.table.get.funcref(ptr addrspace(1) @funcref_table, iX 0)
  ret %funcref %ref
}

define %funcref @get_funcref_from_table_with_offset(iX %i) {
; CHECK-LABEL: get_funcref_from_table_with_offset:
; CHECK-NEXT:   .functype       get_funcref_from_table_with_offset ([[iPTR]]) -> (funcref)
; CHECK-NEXT:   local.get       0
; CHECK-NEXT:   [[iPTR]].const  2
; CHECK-NEXT:   [[iPTR]].add
; CHECK-NEXT:   table.get       funcref_table
; CHECK-NEXT:   end_function
  %off = add nsw iX %i, 2
  %ref = call %funcref @llvm.wasm.table.get.funcref(ptr addrspace(1) @funcref_table, iX %off)
  ret %funcref %ref
}


define %funcref @get_funcref_from_table_with_var_offset(iX %i, iX %j) {
; CHECK-LABEL: get_funcref_from_table_with_var_offset:
; CHECK-NEXT:   .functype       get_funcref_from_table_with_var_offset ([[iPTR]], [[iPTR]]) -> (funcref)
; CHECK-NEXT:   local.get       0
; CHECK-NEXT:   local.get       1
; CHECK-NEXT:   [[iPTR]].add
; CHECK-NEXT:   table.get       funcref_table
; CHECK-NEXT:   end_function
  %off = add nsw iX %i, %j
  %ref = call %funcref @llvm.wasm.table.get.funcref(ptr addrspace(1) @funcref_table, iX %off)
  ret %funcref %ref
}

declare iX @get_offset()

define %funcref @get_funcref_from_table_with_var_offset2(iX %i) {
; CHECK-LABEL: get_funcref_from_table_with_var_offset2:
; CHECK-NEXT:  .functype       get_funcref_from_table_with_var_offset2 ([[iPTR]]) -> (funcref)
; CHECK-NEXT:  local.get       0
; CHECK-NEXT:  call    get_offset
; CHECK-NEXT:  [[iPTR]].add
; CHECK-NEXT:  table.get       funcref_table
; CHECK-NEXT:  end_function
  %j = call iX @get_offset()
  %off = add nsw iX %i, %j
  %ref = call %funcref @llvm.wasm.table.get.funcref(ptr addrspace(1) @funcref_table, iX %off)
  ret %funcref %ref
}

;       CHECK: .tabletype funcref_table, funcref
; CHECK-LABEL: funcref_table:

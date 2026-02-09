; RUN: sed 's/iX/i32/g' < %s | llc --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types | FileCheck %s --check-prefixes=CHECK,WASM32
; RUN: sed 's/iX/i64/g' < %s | llc --mtriple=wasm64-unknown-unknown -asm-verbose=false -mattr=+reference-types | FileCheck %s --check-prefixes=CHECK,WASM64

%funcref = type ptr addrspace(20) ;; addrspace 20 is nonintegral

@funcref_table = local_unnamed_addr addrspace(1) global [0 x %funcref] undef

declare %funcref @llvm.wasm.table.get.funcref(ptr addrspace(1), iX) nounwind

define %funcref @get_funcref_from_table(iX %i) {
; CHECK-LABEL: get_funcref_from_table:
; WASM32-NEXT: .functype       get_funcref_from_table (i32) -> (funcref)
; WASM64-NEXT: .functype       get_funcref_from_table (i64) -> (funcref)
; CHECK-NEXT:  local.get 0
; CHECK-NEXT:  table.get funcref_table
; CHECK-NEXT:  end_function
  %ref = call %funcref @llvm.wasm.table.get.funcref(ptr addrspace(1) @funcref_table, iX %i)
  ret %funcref %ref
}

define %funcref @get_funcref_from_table_const() {
; CHECK-LABEL: get_funcref_from_table_const:
; CHECK-NEXT:  .functype      get_funcref_from_table_const () -> (funcref)
; WASM32-NEXT: i32.const      0
; WASM64-NEXT: i64.const      0
; CHECK-NEXT:  table.get      funcref_table
; CHECK-NEXT:  end_function
  %ref = call %funcref @llvm.wasm.table.get.funcref(ptr addrspace(1) @funcref_table, iX 0)
  ret %funcref %ref
}

define %funcref @get_funcref_from_table_with_offset(iX %i) {
; CHECK-LABEL: get_funcref_from_table_with_offset:
; WASM32-NEXT:  .functype       get_funcref_from_table_with_offset (i32) -> (funcref)
; WASM64-NEXT:  .functype       get_funcref_from_table_with_offset (i64) -> (funcref)
; CHECK-NEXT:   local.get       0
; WASM32-NEXT:  i32.const       2
; WASM32-NEXT:  i32.add
; WASM64-NEXT:  i64.const       2
; WASM64-NEXT:  i64.add
; CHECK-NEXT:   table.get       funcref_table
; CHECK-NEXT:   end_function
  %off = add nsw iX %i, 2
  %ref = call %funcref @llvm.wasm.table.get.funcref(ptr addrspace(1) @funcref_table, iX %off)
  ret %funcref %ref
}


define %funcref @get_funcref_from_table_with_var_offset(iX %i, iX %j) {
; CHECK-LABEL: get_funcref_from_table_with_var_offset:
; WASM32-NEXT:  .functype       get_funcref_from_table_with_var_offset (i32, i32) -> (funcref)
; WASM64-NEXT:  .functype       get_funcref_from_table_with_var_offset (i64, i64) -> (funcref)
; CHECK-NEXT:   local.get       0
; CHECK-NEXT:   local.get       1
; WASM32-NEXT:  i32.add
; WASM64-NEXT:  i64.add
; CHECK-NEXT:   table.get       funcref_table
; CHECK-NEXT:   end_function
  %off = add nsw iX %i, %j
  %ref = call %funcref @llvm.wasm.table.get.funcref(ptr addrspace(1) @funcref_table, iX %off)
  ret %funcref %ref
}

declare iX @get_offset()

define %funcref @get_funcref_from_table_with_var_offset2(iX %i) {
; CHECK-LABEL: get_funcref_from_table_with_var_offset2:
; WASM32-NEXT: .functype       get_funcref_from_table_with_var_offset2 (i32) -> (funcref)
; WASM64-NEXT: .functype       get_funcref_from_table_with_var_offset2 (i64) -> (funcref)
; CHECK-NEXT:  local.get       0
; CHECK-NEXT:  call    get_offset
; WASM32-NEXT: i32.add
; WASM64-NEXT: i64.add
; CHECK-NEXT:  table.get       funcref_table
; CHECK-NEXT:  end_function
  %j = call iX @get_offset()
  %off = add nsw iX %i, %j
  %ref = call %funcref @llvm.wasm.table.get.funcref(ptr addrspace(1) @funcref_table, iX %off)
  ret %funcref %ref
}

;       CHECK: .tabletype funcref_table, funcref
; CHECK-LABEL: funcref_table:

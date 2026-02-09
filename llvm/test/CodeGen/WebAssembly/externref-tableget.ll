; RUN: sed 's/iX/i32/g' < %s | llc --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types | FileCheck %s --check-prefixes=CHECK,WASM32
; RUN: sed 's/iX/i64/g' < %s | llc --mtriple=wasm64-unknown-unknown -asm-verbose=false -mattr=+reference-types | FileCheck %s --check-prefixes=CHECK,WASM64

%externref = type ptr addrspace(10) ;; addrspace 10 is nonintegral

@externref_table = local_unnamed_addr addrspace(1) global [0 x %externref] undef

declare %externref @llvm.wasm.table.get.externref(ptr addrspace(1), iX) nounwind

define %externref @get_externref_from_table(iX %i) {
; CHECK-LABEL: get_externref_from_table:
; WASM32-NEXT: .functype       get_externref_from_table (i32) -> (externref)
; WASM64-NEXT: .functype       get_externref_from_table (i64) -> (externref)
; CHECK-NEXT:  local.get 0
; CHECK-NEXT:  table.get externref_table
; CHECK-NEXT:  end_function
  %ref = call %externref @llvm.wasm.table.get.externref(ptr addrspace(1) @externref_table, iX %i)
  ret %externref %ref
}

define %externref @get_externref_from_table_const() {
; CHECK-LABEL: get_externref_from_table_const:
; CHECK-NEXT:  .functype      get_externref_from_table_const () -> (externref)
; WASM32-NEXT: i32.const      0
; WASM64-NEXT: i64.const      0
; CHECK-NEXT:  table.get      externref_table
; CHECK-NEXT:  end_function
  %ref = call %externref @llvm.wasm.table.get.externref(ptr addrspace(1) @externref_table, iX 0)
  ret %externref %ref
}

define %externref @get_externref_from_table_with_offset(iX %i) {
; CHECK-LABEL: get_externref_from_table_with_offset:
; WASM32-NEXT: .functype       get_externref_from_table_with_offset (i32) -> (externref)
; WASM64-NEXT: .functype       get_externref_from_table_with_offset (i64) -> (externref)
; CHECK-NEXT:  local.get       0
; WASM32-NEXT: i32.const       2
; WASM32-NEXT: i32.add
; WASM64-NEXT: i64.const       2
; WASM64-NEXT: i64.add
; CHECK-NEXT:  table.get       externref_table
; CHECK-NEXT:  end_function
  %off = add nsw iX %i, 2
  %ref = call %externref @llvm.wasm.table.get.externref(ptr addrspace(1) @externref_table, iX %off)
  ret %externref %ref
}


define %externref @get_externref_from_table_with_var_offset(iX %i, iX %j) {
; CHECK-LABEL: get_externref_from_table_with_var_offset:
; WASM32-NEXT: .functype       get_externref_from_table_with_var_offset (i32, i32) -> (externref)
; WASM64-NEXT: .functype       get_externref_from_table_with_var_offset (i64, i64) -> (externref)
; CHECK-NEXT:  local.get       0
; CHECK-NEXT:  local.get       1
; WASM32-NEXT: i32.add
; WASM64-NEXT: i64.add
; CHECK-NEXT:  table.get       externref_table
; CHECK-NEXT:  end_function
  %off = add nsw iX %i, %j
  %ref = call %externref @llvm.wasm.table.get.externref(ptr addrspace(1) @externref_table, iX %off)
  ret %externref %ref
}

declare iX @get_offset()

define %externref @get_externref_from_table_with_var_offset2(iX %i) {
; CHECK-LABEL: get_externref_from_table_with_var_offset2:
; WASM32-NEXT: .functype       get_externref_from_table_with_var_offset2 (i32) -> (externref)
; WASM64-NEXT: .functype       get_externref_from_table_with_var_offset2 (i64) -> (externref)
; CHECK-NEXT:  local.get       0
; CHECK-NEXT:  call    get_offset
; WASM32-NEXT: i32.add
; WASM64-NEXT: i64.add
; CHECK-NEXT:  table.get       externref_table
; CHECK-NEXT:  end_function
  %j = call iX @get_offset()
  %off = add nsw iX %i, %j
  %ref = call %externref @llvm.wasm.table.get.externref(ptr addrspace(1) @externref_table, iX %off)
  ret %externref %ref
}

; CHECK:       .tabletype externref_table, externref
; CHECK-LABEL: externref_table:

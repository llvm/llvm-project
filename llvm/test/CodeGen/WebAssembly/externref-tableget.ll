; RUN: sed 's/iX/i32/g' < %s > %t && llc --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types < %t | FileCheck %t
; RUN: sed 's/iX/i64/g' < %s > %t && llc --mtriple=wasm64-unknown-unknown -asm-verbose=false -mattr=+reference-types < %t | FileCheck %t

%externref = type target("wasm.externref")

@externref_table = local_unnamed_addr addrspace(1) global [0 x %externref] undef

declare %externref @llvm.wasm.table.get.externref(ptr addrspace(1), iX) nounwind

define %externref @get_externref_from_table(iX %i) {
; CHECK-LABEL: get_externref_from_table:
; CHECK-NEXT:  .functype       get_externref_from_table (iX) -> (externref)
; CHECK-NEXT:  local.get 0
; CHECK-NEXT:  table.get externref_table
; CHECK-NEXT:  end_function
  %ref = call %externref @llvm.wasm.table.get.externref(ptr addrspace(1) @externref_table, iX %i)
  ret %externref %ref
}

define %externref @get_externref_from_table_const() {
; CHECK-LABEL: get_externref_from_table_const:
; CHECK-NEXT:  .functype      get_externref_from_table_const () -> (externref)
; CHECK-NEXT:  iX.const 0
; CHECK-NEXT:  table.get      externref_table
; CHECK-NEXT:  end_function
  %ref = call %externref @llvm.wasm.table.get.externref(ptr addrspace(1) @externref_table, iX 0)
  ret %externref %ref
}

define %externref @get_externref_from_table_with_offset(iX %i) {
; CHECK-LABEL: get_externref_from_table_with_offset:
; CHECK-NEXT:  .functype       get_externref_from_table_with_offset (iX) -> (externref)
; CHECK-NEXT:  local.get       0
; CHECK-NEXT:  iX.const  2
; CHECK-NEXT:  iX.add
; CHECK-NEXT:  table.get       externref_table
; CHECK-NEXT:  end_function
  %off = add nsw iX %i, 2
  %ref = call %externref @llvm.wasm.table.get.externref(ptr addrspace(1) @externref_table, iX %off)
  ret %externref %ref
}


define %externref @get_externref_from_table_with_var_offset(iX %i, iX %j) {
; CHECK-LABEL: get_externref_from_table_with_var_offset:
; CHECK-NEXT:  .functype       get_externref_from_table_with_var_offset (iX, iX) -> (externref)
; CHECK-NEXT:  local.get       0
; CHECK-NEXT:  local.get       1
; CHECK-NEXT:  iX.add
; CHECK-NEXT:  table.get       externref_table
; CHECK-NEXT:  end_function
  %off = add nsw iX %i, %j
  %ref = call %externref @llvm.wasm.table.get.externref(ptr addrspace(1) @externref_table, iX %off)
  ret %externref %ref
}

declare iX @get_offset()

define %externref @get_externref_from_table_with_var_offset2(iX %i) {
; CHECK-LABEL: get_externref_from_table_with_var_offset2:
; CHECK-NEXT: .functype       get_externref_from_table_with_var_offset2 (iX) -> (externref)
; CHECK-NEXT:  local.get      0
; CHECK-NEXT:  call    get_offset
; CHECK-NEXT:  iX.add
; CHECK-NEXT:  table.get       externref_table
; CHECK-NEXT:  end_function
  %j = call iX @get_offset()
  %off = add nsw iX %i, %j
  %ref = call %externref @llvm.wasm.table.get.externref(ptr addrspace(1) @externref_table, iX %off)
  ret %externref %ref
}

; CHECK:       .tabletype externref_table, externref
; CHECK-LABEL: externref_table:

; RUN: sed 's/iX/i32/g' < %s > %t && llc --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types < %t | FileCheck %t
; RUN: sed 's/iX/i64/g' < %s > %t && llc --mtriple=wasm64-unknown-unknown -asm-verbose=false -mattr=+reference-types < %t | FileCheck %t

%externref = type target("wasm.externref")

@externref_table = local_unnamed_addr addrspace(1) global [0 x %externref] undef

declare void @llvm.wasm.table.fill.externref(ptr addrspace(1), iX, %externref, iX) nounwind readonly

define void @table_fill(iX %start, iX %len, %externref %val) {
; CHECK-LABEL: table_fill:
; CHECK-NEXT:  .functype	table_fill (iX, iX, externref) -> ()
; CHECK-NEXT:  local.get    0
; CHECK-NEXT:  local.get    2
; CHECK-NEXT:  local.get    1
; CHECK-NEXT:  table.fill	externref_table
; CHECK-NEXT:  end_function
  call void @llvm.wasm.table.fill.externref(ptr addrspace(1) @externref_table, iX %start, %externref %val, iX %len)
  ret void
}

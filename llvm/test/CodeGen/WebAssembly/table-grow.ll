; RUN: sed 's/iX/i32/g' < %s > %t && llc --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types < %t | FileCheck %t
; RUN: sed 's/iX/i64/g' < %s > %t && llc --mtriple=wasm64-unknown-unknown -asm-verbose=false -mattr=+reference-types < %t | FileCheck %t

%externref = type target("wasm.externref")

@externref_table = local_unnamed_addr addrspace(1) global [0 x %externref] undef

declare iX @llvm.wasm.table.grow.externref(ptr addrspace(1), %externref, iX) nounwind readonly
declare %externref @llvm.wasm.ref.null.extern() nounwind readonly

define iX @table_grow(iX %sz) {
; CHECK-LABEL: table_grow:
; CHECK-NEXT:  .functype	table_grow (iX) -> (iX)
; CHECK-NEXT:  ref.null_extern
; CHECK-NEXT:  local.get	0
; CHECK-NEXT:  table.grow	externref_table
; CHECK-NEXT:  end_function
  %null = call %externref @llvm.wasm.ref.null.extern()
  %newsz = call iX @llvm.wasm.table.grow.externref(ptr addrspace(1) @externref_table, %externref %null, iX %sz)
  ret iX %newsz
}

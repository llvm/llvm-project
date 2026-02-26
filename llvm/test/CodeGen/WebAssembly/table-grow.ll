; RUN: sed 's/iX/i32/g' < %s | llc --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types | FileCheck %s -DiPTR=i32
; RUN: sed 's/iX/i64/g' < %s | llc --mtriple=wasm64-unknown-unknown -asm-verbose=false -mattr=+reference-types | FileCheck %s -DiPTR=i64

%externref = type ptr addrspace(10) ;; addrspace 10 is nonintegral

@externref_table = local_unnamed_addr addrspace(1) global [0 x %externref] undef

declare i32 @llvm.wasm.table.grow.externref(ptr addrspace(1), %externref, iX) nounwind readonly
declare %externref @llvm.wasm.ref.null.extern() nounwind readonly

define i32 @table_grow(iX %sz) {
; CHECK-LABEL: table_grow:
; CHECK-NEXT:  .functype	table_grow ([[iPTR]]) -> (i32)
; CHECK-NEXT:  ref.null_extern
; CHECK-NEXT:  local.get	0
; CHECK-NEXT:  table.grow	externref_table
; CHECK-NEXT:  end_function
  %null = call %externref @llvm.wasm.ref.null.extern()
  %newsz = call i32 @llvm.wasm.table.grow.externref(ptr addrspace(1) @externref_table, %externref %null, iX %sz)
  ret i32 %newsz
}

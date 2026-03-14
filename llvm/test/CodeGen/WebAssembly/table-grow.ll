; RUN: llc --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types < %s | FileCheck %s

%externref = type ptr addrspace(10) ;; addrspace 10 is nonintegral

@externref_table = local_unnamed_addr addrspace(1) global [0 x %externref] undef

declare i32 @llvm.wasm.table.grow.externref(ptr addrspace(1), %externref, i32) nounwind readonly
declare %externref @llvm.wasm.ref.null.extern() nounwind readonly

define i32 @table_grow(i32 %sz) {
; CHECK-LABEL: table_grow:
; CHECK-NEXT:  .functype	table_grow (i32) -> (i32)
; CHECK-NEXT:  ref.null_extern
; CHECK-NEXT:  local.get	0
; CHECK-NEXT:  table.grow	externref_table
; CHECK-NEXT:  end_function
  %null = call %externref @llvm.wasm.ref.null.extern()
  %newsz = call i32 @llvm.wasm.table.grow.externref(ptr addrspace(1) @externref_table, %externref %null, i32 %sz)
  ret i32 %newsz
}

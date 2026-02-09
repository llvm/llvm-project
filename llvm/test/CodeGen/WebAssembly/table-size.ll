; RUN: sed 's/iX/i32/g' < %s | llc --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types | FileCheck %s --check-prefixes=CHECK,WASM32
; RUN: sed 's/iX/i64/g' < %s | llc --mtriple=wasm64-unknown-unknown -asm-verbose=false -mattr=+reference-types | FileCheck %s --check-prefixes=CHECK,WASM64

%externref = type ptr addrspace(10) ;; addrspace 10 is nonintegral

@externref_table = local_unnamed_addr addrspace(1) global [0 x %externref] undef

declare iX @llvm.wasm.table.size(ptr addrspace(1)) nounwind readonly

define iX @table_size() {
; CHECK-LABEL: table_size:
; WASM32-NEXT: .functype       table_size () -> (i32)
; WASM64-NEXT: .functype       table_size () -> (i64)
; CHECK-NEXT:  table.size      externref_table
; CHECK-NEXT:  end_function
  %sz = call iX @llvm.wasm.table.size(ptr addrspace(1) @externref_table)
  ret iX %sz
}

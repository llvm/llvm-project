; RUN: sed 's/iX/i32/g' < %s | llc --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types | FileCheck %s --check-prefixes=CHECK,WASM32
; RUN: sed 's/iX/i64/g' < %s | llc --mtriple=wasm64-unknown-unknown -asm-verbose=false -mattr=+reference-types | FileCheck %s --check-prefixes=CHECK,WASM64

%externref = type ptr addrspace(10) ;; addrspace 10 is nonintegral

@externref_table1 = local_unnamed_addr addrspace(1) global [0 x %externref] undef
@externref_table2 = local_unnamed_addr addrspace(1) global [0 x %externref] undef

declare void @llvm.wasm.table.copy(ptr addrspace(1), ptr addrspace(1), iX, iX, iX) nounwind readonly

define void @table_copy(iX %dst, iX %src, iX %len) {
; CHECK-LABEL: table_copy:
; WASM32-NEXT: .functype	table_copy (i32, i32, i32) -> ()
; WASM64-NEXT: .functype	table_copy (i64, i64, i64) -> ()
; CHECK-NEXT:  local.get    0
; CHECK-NEXT:  local.get    1
; CHECK-NEXT:  local.get    2
; CHECK-NEXT:  table.copy	externref_table1, externref_table2
; CHECK-NEXT:  end_function
  call void @llvm.wasm.table.copy(ptr addrspace(1) @externref_table1, ptr addrspace(1) @externref_table2, iX %dst, iX %src, iX %len)
  ret void
}

; Testing copying from a table to itself at different offsets
; Copies len items from table1 at src to table1 at src+off
define void @self_table_copy(iX %src, iX %off, iX %len) {
; CHECK-LABEL: self_table_copy:
; WASM32-NEXT: .functype	self_table_copy (i32, i32, i32) -> ()
; WASM64-NEXT: .functype	self_table_copy (i64, i64, i64) -> ()
; CHECK-NEXT:  local.get    0
; CHECK-NEXT:  local.get    1
; WASM32-NEXT: i32.add
; WASM64-NEXT: i64.add
; CHECK-NEXT:  local.get    0
; CHECK-NEXT:  local.get    2
; CHECK-NEXT:  table.copy	externref_table1, externref_table1
; CHECK-NEXT:  end_function
  %dst = add nsw iX %src, %off
  call void @llvm.wasm.table.copy(ptr addrspace(1) @externref_table1, ptr addrspace(1) @externref_table1, iX %dst, iX %src, iX %len)
  ret void
}

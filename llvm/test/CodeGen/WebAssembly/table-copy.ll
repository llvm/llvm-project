; RUN: llc --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types < %s | FileCheck %s

%externref = type ptr addrspace(10) ;; addrspace 10 is nonintegral

@externref_table1 = local_unnamed_addr addrspace(1) global [0 x %externref] undef
@externref_table2 = local_unnamed_addr addrspace(1) global [0 x %externref] undef

declare void @llvm.wasm.table.copy(ptr addrspace(1), ptr addrspace(1), i32, i32, i32) nounwind readonly

define void @table_copy(i32 %dst, i32 %src, i32 %len) {
; CHECK-LABEL: table_copy:
; CHECK-NEXT:  .functype	table_copy (i32, i32, i32) -> ()
; CHECK-NEXT:  local.get    0
; CHECK-NEXT:  local.get    1
; CHECK-NEXT:  local.get    2
; CHECK-NEXT:  table.copy	externref_table1, externref_table2
; CHECK-NEXT:  end_function
  call void @llvm.wasm.table.copy(ptr addrspace(1) @externref_table1, ptr addrspace(1) @externref_table2, i32 %dst, i32 %src, i32 %len)
  ret void
}

; Testing copying from a table to itself at different offsets
; Copies len items from table1 at src to table1 at src+off
define void @self_table_copy(i32 %src, i32 %off, i32 %len) {
; CHECK-LABEL: self_table_copy:
; CHECK-NEXT:  .functype	self_table_copy (i32, i32, i32) -> ()
; CHECK-NEXT:  local.get    0
; CHECK-NEXT:  local.get    1
; CHECK-NEXT:  i32.add
; CHECK-NEXT:  local.get    0
; CHECK-NEXT:  local.get    2
; CHECK-NEXT:  table.copy	externref_table1, externref_table1
; CHECK-NEXT:  end_function
  %dst = add nsw i32 %src, %off
  call void @llvm.wasm.table.copy(ptr addrspace(1) @externref_table1, ptr addrspace(1) @externref_table1, i32 %dst, i32 %src, i32 %len)
  ret void
}

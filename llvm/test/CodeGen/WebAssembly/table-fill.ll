; RUN: llc --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types < %s | FileCheck %s

%extern = type opaque
%externref = type %extern addrspace(10)* ;; addrspace 10 is nonintegral

@externref_table = local_unnamed_addr addrspace(1) global [0 x %externref] undef

declare void @llvm.wasm.table.fill.externref(i8 addrspace(1)*, i32, %externref, i32) nounwind readonly

define void @table_fill(i32 %start, i32 %len, %externref %val) {
; CHECK-LABEL: table_fill:
; CHECK-NEXT:  .functype	table_fill (i32, i32, externref) -> ()
; CHECK-NEXT:  local.get    0
; CHECK-NEXT:  local.get    2
; CHECK-NEXT:  local.get    1
; CHECK-NEXT:  table.fill	externref_table
; CHECK-NEXT:  end_function
  %tableptr = getelementptr [0 x %externref], [0 x %externref] addrspace(1)* @externref_table, i32 0, i32 0
  %tb = bitcast %externref addrspace(1)* %tableptr to i8 addrspace(1)*
  call void @llvm.wasm.table.fill.externref(i8 addrspace(1)* %tb, i32 %start, %externref %val, i32 %len)
  ret void
}

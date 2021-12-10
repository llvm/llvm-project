; RUN: llc --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types < %s | FileCheck %s

%extern = type opaque
%externref = type %extern addrspace(10)* ;; addrspace 10 is nonintegral

@externref_table = local_unnamed_addr addrspace(1) global [0 x %externref] undef

declare i32 @llvm.wasm.table.size(i8 addrspace(1)*) nounwind readonly

define i32 @table_size() {
; CHECK-LABEL: table_size:
; CHECK-NEXT:  .functype       table_size () -> (i32)
; CHECK-NEXT:  table.size      externref_table
; CHECK-NEXT:  end_function
  %tableptr = getelementptr [0 x %externref], [0 x %externref] addrspace(1)* @externref_table, i32 0, i32 0
  %tb = bitcast %externref addrspace(1)* %tableptr to i8 addrspace(1)*
  %sz = call i32 @llvm.wasm.table.size(i8 addrspace(1)* %tb)
  ret i32 %sz
}

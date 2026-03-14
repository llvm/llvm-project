; RUN: llc < %s --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types | FileCheck %s

%funcref = type ptr addrspace(20) ;; addrspace 20 is nonintegral

@funcref_table = local_unnamed_addr addrspace(1) global [0 x %funcref] undef

;  CHECK: .tabletype  __funcref_call_table, funcref, 1

declare %funcref @llvm.wasm.table.get.funcref(ptr addrspace(1), i32) nounwind

define void @call_funcref_from_table(i32 %i) {
; CHECK-LABEL: call_funcref_from_table:
; CHECK-NEXT: .functype       call_funcref_from_table (i32) -> ()
; CHECK-NEXT: i32.const       0
; CHECK-NEXT: local.get       0
; CHECK-NEXT: table.get       funcref_table
; CHECK-NEXT: table.set       __funcref_call_table
; CHECK-NEXT: i32.const       0
; CHECK-NEXT: call_indirect    __funcref_call_table, () -> ()
; CHECK-NEXT: i32.const       0
; CHECK-NEXT: ref.null_func
; CHECK-NEXT: table.set       __funcref_call_table
; CHECK-NEXT: end_function
  %ref = call %funcref @llvm.wasm.table.get.funcref(ptr addrspace(1) @funcref_table, i32 %i)
  call addrspace(20) void %ref()
  ret void
}

;       CHECK: .tabletype funcref_table, funcref
; CHECK-LABEL: funcref_table:


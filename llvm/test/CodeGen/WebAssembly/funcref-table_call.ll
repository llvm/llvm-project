; RUN: sed 's/iX/i32/g' < %s | llc --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types | FileCheck %s --check-prefixes=CHECK,WASM32
; RUN: sed 's/iX/i64/g' < %s | llc --mtriple=wasm64-unknown-unknown -asm-verbose=false -mattr=+reference-types | FileCheck %s --check-prefixes=CHECK,WASM64

%funcref = type target("wasm.funcref")

@funcref_table = local_unnamed_addr addrspace(1) global [0 x %funcref] undef

;  CHECK: .tabletype  __funcref_call_table, funcref, 1

declare %funcref @llvm.wasm.table.get.funcref(ptr addrspace(1), iX) nounwind
declare ptr @llvm.wasm.funcref.to_ptr(%funcref) nounwind

define void @call_funcref_from_table(iX %i) {
; CHECK-LABEL: call_funcref_from_table:
; WASM32-NEXT: .functype       call_funcref_from_table (i32) -> ()
; WASM64-NEXT: .functype       call_funcref_from_table (i64) -> ()
; WASM32-NEXT: i32.const       0
; WASM64-NEXT: i64.const       0
; CHECK-NEXT:  local.get       0
; CHECK-NEXT:  table.get       funcref_table
; CHECK-NEXT:  table.set       __funcref_call_table
; WASM32-NEXT: i32.const       0
; WASM64-NEXT: i64.const       0
; CHECK-NEXT:  call_indirect    __funcref_call_table, () -> ()
; WASM32-NEXT: i32.const       0
; WASM64-NEXT: i64.const       0
; CHECK-NEXT:  ref.null_func
; CHECK-NEXT:  table.set       __funcref_call_table
; CHECK-NEXT:  end_function
  %ref = call %funcref @llvm.wasm.table.get.funcref(ptr addrspace(1) @funcref_table, iX %i)
  %refptr = call ptr @llvm.wasm.funcref.to_ptr(%funcref %ref)
  call void %refptr()
  ret void
}

;       CHECK: .tabletype funcref_table, funcref
; CHECK-LABEL: funcref_table:

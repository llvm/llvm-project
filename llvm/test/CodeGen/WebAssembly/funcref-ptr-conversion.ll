; RUN: llc < %s --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types | FileCheck %s --check-prefixes=CHECK,WASM32
; RUN: llc < %s --mtriple=wasm64-unknown-unknown -asm-verbose=false -mattr=+reference-types | FileCheck %s --check-prefixes=CHECK,WASM64

%funcref = type target("wasm.funcref")

declare %funcref @llvm.wasm.ptr.to_funcref(ptr) nounwind

; CHECK: .tabletype __indirect_function_table, funcref

; Converting a function pointer to a funcref is a table.get from the
; __indirect_function_table.
define %funcref @ptr_to_funcref(ptr %p) {
; CHECK-LABEL: ptr_to_funcref:
; WASM32:         .functype ptr_to_funcref (i32) -> (funcref)
; WASM64:         .functype ptr_to_funcref (i64) -> (funcref)
; CHECK-NEXT:    local.get 0
; CHECK-NEXT:    table.get __indirect_function_table
; CHECK-NEXT:    end_function
  %ref = call %funcref @llvm.wasm.ptr.to_funcref(ptr %p)
  ret %funcref %ref
}

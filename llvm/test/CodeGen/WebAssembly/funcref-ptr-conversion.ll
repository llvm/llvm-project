; RUN: llc < %s --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types | FileCheck %s
; RUN: llc < %s --mtriple=wasm64-unknown-unknown -asm-verbose=false -mattr=+reference-types | FileCheck %s --check-prefix=CHECK64

%funcref = type target("wasm.funcref")

declare %funcref @llvm.wasm.ptr.to_funcref(ptr) nounwind

; CHECK: .tabletype __indirect_function_table, funcref

; Converting a function pointer to a funcref is a table.get from the
; __indirect_function_table.
define %funcref @ptr_to_funcref(ptr %p) {
; CHECK-LABEL: ptr_to_funcref:
; CHECK:         .functype ptr_to_funcref (i32) -> (funcref)
; CHECK-NEXT:    local.get 0
; CHECK-NEXT:    table.get __indirect_function_table
; CHECK-NEXT:    end_function
;
; On wasm64 the function pointer is an i64 and must be wrapped to i32 first.
; CHECK64-LABEL: ptr_to_funcref:
; CHECK64:         .functype ptr_to_funcref (i64) -> (funcref)
; CHECK64-NEXT:    local.get 0
; CHECK64-NEXT:    i32.wrap_i64
; CHECK64-NEXT:    table.get __indirect_function_table
; CHECK64-NEXT:    end_function
  %ref = call %funcref @llvm.wasm.ptr.to_funcref(ptr %p)
  ret %funcref %ref
}

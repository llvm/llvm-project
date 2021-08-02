; RUN: llc < %s --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types | FileCheck %s

%func = type void ()
%funcref = type %func addrspace(20)* ;; addrspace 20 is nonintegral

define void @call_funcref(%funcref %ref) {
  call addrspace(20) void %ref() 
  ret void
}

; CHECK: .tabletype __funcref_call_table, funcref, 1

; CHECK-LABEL: call_funcref:
; CHECK-NEXT: functype       call_funcref (funcref) -> ()
; CHECK-NEXT: i32.const 0
; CHECK-NEXT: local.get 0
; CHECK-NEXT: table.set __funcref_call_table
; CHECK-NEXT: local.get 0
; CHECK-NEXT: call_indirect __funcref_call_table, () -> ()
; CHECK-NEXT: i32.const 0
; CHECK-NEXT: ref.null func
; CHECK-NEXT: table.set __funcref_call_table
; CHECK-NEXT: end_function


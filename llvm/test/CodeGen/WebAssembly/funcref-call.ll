; RUN: llc < %s --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types | FileCheck %s

%funcptr = type void () addrspace(20)*
%funcref = type i8 addrspace(20)* ;; addrspace 20 is nonintegral

; CHECK: .tabletype __funcref_call_table, funcref, 1

define void @call_funcref(%funcref %ref) {
; CHECK-LABEL: call_funcref:
; CHECK-NEXT: .functype       call_funcref (funcref) -> ()
; CHECK-NEXT: i32.const 0
; CHECK-NEXT: local.get 0
; CHECK-NEXT: table.set __funcref_call_table
; CHECK-NEXT: i32.const 0
; CHECK-NEXT: call_indirect __funcref_call_table, () -> ()
; CHECK-NEXT: i32.const 0
; CHECK-NEXT: ref.null_func
; CHECK-NEXT: table.set __funcref_call_table
; CHECK-NEXT: end_function
  %f = bitcast %funcref %ref to %funcptr
  call addrspace(20) void %f() 
  ret void
}

define void @call_funcptr(%funcptr %ref) {
; CHECK-LABEL: call_funcptr:
; CHECK-NEXT: .functype       call_funcptr (funcref) -> ()
; CHECK-NEXT: i32.const	0
; CHECK-NEXT: local.get	0
; CHECK-NEXT: table.set	__funcref_call_table
; CHECK-NEXT: i32.const	0
; CHECK-NEXT: call_indirect	__funcref_call_table, () -> ()
; CHECK-NEXT: i32.const	0
; CHECK-NEXT: ref.null_func
; CHECK-NEXT: table.set	__funcref_call_table
; CHECK-NEXT: end_function
  call addrspace(20) void %ref()
  ret void
}




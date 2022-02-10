; RUN: llc < %s --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types | FileCheck %s

%funcptr = type void () addrspace(20)*
%funcref = type i8 addrspace(20)* ;; addrspace 20 is nonintegral

@funcref_table = local_unnamed_addr addrspace(1) global [0 x %funcref] undef

;  CHECK: .tabletype  __funcref_call_table, funcref, 1

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
  %p = getelementptr [0 x %funcref], [0 x %funcref] addrspace (1)* @funcref_table, i32 0, i32 %i
  %ref = load %funcref, %funcref addrspace(1)* %p
  %fn = bitcast %funcref %ref to %funcptr
  call addrspace(20) void %fn()
  ret void
}

;       CHECK: .tabletype funcref_table, funcref
; CHECK-LABEL: funcref_table:


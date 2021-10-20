; RUN: llc < %s --mtriple=wasm32-unknown-unknown -asm-verbose=false -mattr=+reference-types | FileCheck %s

%func = type void ()
%funcref = type %func addrspace(20)* ;; addrspace 20 is nonintegral

@funcref_table = local_unnamed_addr addrspace(1) global [0 x %funcref] undef

define %funcref @get_funcref_from_table(i32 %i) {
; CHECK-LABEL: get_funcref_from_table:
; CHECK-NEXT: .functype       get_funcref_from_table (i32) -> (funcref)
; CHECK-NEXT: local.get 0
; CHECK-NEXT: table.get funcref_table
; CHECK-NEXT: end_function
  %p = getelementptr [0 x %funcref], [0 x %funcref] addrspace (1)* @funcref_table, i32 0, i32 %i
  %ref = load %funcref, %funcref addrspace(1)* %p
  ret %funcref %ref
}

define %funcref @get_funcref_from_table_const() {
; CHECK-LABEL: get_funcref_from_table_const:
; CHECK-NEXT:  .functype      get_funcref_from_table_const () -> (funcref)
; CHECK-NEXT:  i32.const      0
; CHECK-NEXT:  table.get      funcref_table
; CHECK-NEXT:  end_function
  %p = getelementptr [0 x %funcref], [0 x %funcref] addrspace (1)* @funcref_table, i32 0, i32 0
  %ref = load %funcref, %funcref addrspace(1)* %p
  ret %funcref %ref
}

define %funcref @get_funcref_from_table_with_offset(i32 %i) {
; CHECK-LABEL: get_funcref_from_table_with_offset:
; CHECK-NEXT:  .functype       get_funcref_from_table_with_offset (i32) -> (funcref)
; CHECK-NEXT:  local.get       0
; CHECK-NEXT:  i32.const       2
; CHECK-NEXT:  i32.add
; CHECK-NEXT:  table.get       funcref_table
; CHECK-NEXT:  end_function
  %off = add nsw i32 %i, 2
  %p = getelementptr [0 x %funcref], [0 x %funcref] addrspace (1)* @funcref_table, i32 0, i32 %off
  %ref = load %funcref, %funcref addrspace(1)* %p
  ret %funcref %ref
}


define %funcref @get_funcref_from_table_with_var_offset(i32 %i, i32 %j) {
; CHECK-LABEL: get_funcref_from_table_with_var_offset:
; CHECK-NEXT:  .functype       get_funcref_from_table_with_var_offset (i32, i32) -> (funcref)
; CHECK-NEXT:  local.get       0
; CHECK-NEXT:  local.get       1
; CHECK-NEXT:  i32.add
; CHECK-NEXT:  table.get       funcref_table
; CHECK-NEXT:  end_function
  %off = add nsw i32 %i, %j
  %p = getelementptr [0 x %funcref], [0 x %funcref] addrspace (1)* @funcref_table, i32 0, i32 %off
  %ref = load %funcref, %funcref addrspace(1)* %p
  ret %funcref %ref
}

declare i32 @get_offset()

define %funcref @get_funcref_from_table_with_var_offset2(i32 %i) {
; CHECK-LABEL: get_funcref_from_table_with_var_offset2:
; CHECK-NEXT:  .functype       get_funcref_from_table_with_var_offset2 (i32) -> (funcref)
; CHECK-NEXT:  local.get       0
; CHECK-NEXT:  call    get_offset
; CHECK-NEXT:  i32.add
; CHECK-NEXT:  table.get       funcref_table
; CHECK-NEXT:  end_function
  %j = call i32 @get_offset()
  %off = add nsw i32 %i, %j
  %p = getelementptr [0 x %funcref], [0 x %funcref] addrspace (1)* @funcref_table, i32 0, i32 %off
  %ref = load %funcref, %funcref addrspace(1)* %p
  ret %funcref %ref
}

; CHECK: .globl funcref_table

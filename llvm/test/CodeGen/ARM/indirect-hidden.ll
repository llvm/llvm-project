; RUN: llc -mtriple=thumbv7s-apple-ios7.0 -o - %s | FileCheck %s

@var = external global i32
@var_hidden = external hidden global i32

define i32* @get_var() {
  ret i32* @var
}

define i32* @get_var_hidden() {
  ret i32* @var_hidden
}

; CHECK: .section __DATA,__nl_symbol_ptr,non_lazy_symbol_pointers

; CHECK: .indirect_symbol _var
; CHECK-NEXT: .long 0

; CHECK-NOT: __DATA,__data

; CHECK: .indirect_symbol _var_hidden
; CHECK-NEXT: .long 0

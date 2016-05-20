; RUN: llc -mtriple=powerpc-apple-darwin < %s | FileCheck %s

@a = external hidden global i32
@b = external global i32

define i32* @get_a() {
  ret i32* @a
}

define i32* @get_b() {
  ret i32* @b
}

; CHECK:      .section __DATA,__nl_symbol_ptr,non_lazy_symbol_pointers
; CHECK-NEXT: .p2align  2
; CHECK-NEXT: L_a$non_lazy_ptr:
; CHECK-NEXT:   .indirect_symbol        _a
; CHECK-NEXT:   .long   0
; CHECK-NEXT: L_b$non_lazy_ptr:
; CHECK-NEXT:   .indirect_symbol        _b
; CHECK-NEXT:   .long   0

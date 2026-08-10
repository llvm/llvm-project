; RUN: llc < %s -mtriple=i686-pc-windows-msvc -relocation-model=pic | FileCheck %s
; MOVPC32r should not generate CFI under windows

; CHECK-LABEL: _foo:
; CHECK-NOT: .cfi_adjust_cfa_offset
define void @foo(i8) {
entry-block:
  switch i8 %0, label %bb2 [
    i8 1, label %bb1
    i8 2, label %bb2
    i8 3, label %bb3
    i8 4, label %bb4
    i8 5, label %bb5
  ]

bb1:
  ret void

bb2:
  ret void

bb3:
  ret void

bb4:
  ret void

bb5:
  ret void
}

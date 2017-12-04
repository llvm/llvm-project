; Ensure that adjacent duplicated barriers are not removed at -O0.
; RUN: llc -O0 < %s -mtriple=armv7 -mattr=+db | FileCheck %s

define i32 @t1() {
entry:
  fence seq_cst
  fence seq_cst
  fence seq_cst
  ret i32 0
}

; CHECK: @ %bb.0: @ %entry
; CHECK-NEXT: dmb ish
; CHECK-NEXT: dmb ish
; CHECK-NEXT: dmb ish

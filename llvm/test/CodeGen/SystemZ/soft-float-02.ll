; RUN: llc < %s -mtriple=s390x-linux-gnu -mattr=soft-float | FileCheck %s
;
; Check that FP registers are not saved in a vararg function if soft-float is
; used.

define void @fun0(...) {
; CHECK-LABEL: fun0
; CHECK-NOT: std %f0
; CHECK-NOT: std %f2
; CHECK-NOT: std %f4
; CHECK-NOT: std %f6
  ret void
}



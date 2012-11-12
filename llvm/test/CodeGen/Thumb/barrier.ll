; RUN: llc < %s -mtriple=thumbv6-apple-darwin  | FileCheck %s -check-prefix=V6
; RUN: llc < %s -mtriple=thumbv7-apple-darwin -mattr=-db | FileCheck %s -check-prefix=V6
; RUN: llc < %s -march=thumb -mcpu=cortex-m0   | FileCheck %s -check-prefix=V6M

define void @t1() {
; V6: t1:
; V6: blx {{_*}}sync_synchronize

; V6M: t1:
; V6M: dmb ish
  fence seq_cst
  ret void
}

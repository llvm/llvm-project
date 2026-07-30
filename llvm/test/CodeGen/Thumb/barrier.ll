; RUN: llc -mtriple=thumbv6-apple-darwin %s -o - | FileCheck %s -check-prefix=V6
; RUN: llc -mtriple=thumbv7-apple-darwin -mattr=-db %s -o - | FileCheck %s -check-prefix=V6
; RUN: llc -mtriple=thumb-eabi -mcpu=cortex-m0 %s -o - | FileCheck %s -check-prefix=V6M

define void @t1() {
; V6-LABEL: t1:
; V6: bl {{_*}}sync_synchronize

; V6M-LABEL: t1:
; V6M: dmb sy
  fence seq_cst
  ret void
}

; RUN: llc -mtriple s390x-ibm-zos -mcpu=z15 -asm-verbose=true < %s | FileCheck %s
; REQUIRES: systemz-registered-target

; CHECK:    .section    ".ppa2"
; CHECK: @@PPA2:
; CHECK:    .byte   3
; CHECK:    .byte   231
; CHECK:    .byte   34
; CHECK:    .byte   4
; CHECK:    .long   CELQSTRT-@@PPA2
; CHECK:    .long   0
; CHECK:    .long   @@DVS-@@PPA2
; CHECK:    .long   0
; CHECK:    .byte   129
; CHECK:    .byte   0
; CHECK:    .short  0
; CHECK: @@DVS:
; CHECK:    .ascii  "\361\371\367\360\360\361\360\361\360\360\360\360\360\360"
; CHECK:    .short  0
; CHECK:    .quad   @@PPA2-CELQSTRT                 * A(PPA2-CELQSTRT)
; CHECK: @@PPA1_void_test_0:
; CHECK:    .long   @@PPA2-@@PPA1_void_test_0       * Offset to PPA2
; CHECK:    .section    "B_IDRL"
; CHECK:    .byte   0
; CHECK:    .byte   3
; CHECK:    .short  30
; CHECK:    .ascii  "\323\323\345\324@@@@@@\361\370\360\360\361\371\367\360\360\361\360\361\360\360\360\360\360\360\360\360"
define void @void_test() {
entry:
  ret void
}

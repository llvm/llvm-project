; RUN: llc -mtriple s390x-ibm-zos -mcpu=z15 -asm-verbose=true < %s | FileCheck %s

; CHECK:         .section    ".text"
; CHECK-NEXT: L#PPA2:
; CHECK-NEXT:    .byte   3
; CHECK-NEXT:    .byte   231
; CHECK-NEXT:    .byte   34
; CHECK-NEXT:    .byte   4
; CHECK-NEXT:    .long   CELQSTRT-L#PPA2
; CHECK-NEXT:    .long   0
; CHECK-NEXT:    .long   L#DVS-L#PPA2
; CHECK-NEXT:    .long   0
; CHECK-NEXT:    .byte   129
; CHECK-NEXT:    .byte   0
; CHECK-NEXT:    .short  0
; CHECK-NEXT: L#DVS:
; CHECK-NEXT:    .ascii  "\361\371\367\360\360\361\360\361\360\360\360\360\360\360"
; CHECK-NEXT:    .ascii	 "\362\360\360\360\360\360"
; CHECK-NEXT:    .short  0
; CHECK-NEXT:	   .section	".ppa2list"
; CHECK-NEXT:    .quad   L#PPA2-CELQSTRT                 * A(PPA2-CELQSTRT)
; CHECK: L#PPA1_void_test_0:
; CHECK:    .long   L#PPA2-L#PPA1_void_test_0       * Offset to PPA2
; CHECK:    .section    "B_IDRL"
; CHECK:    .byte   0
; CHECK:    .byte   3
; CHECK:    .short  30
; CHECK:    .ascii  "\323\323\345\324@@@@@@{{((\\3[0-7]{2}){4})}}\361\371\367\360\360\361\360\361\360\360\360\360\360\360\360\360"
define void @void_test() {
entry:
  ret void
}

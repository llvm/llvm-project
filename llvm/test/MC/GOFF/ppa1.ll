; RUN: llc -mtriple s390x-ibm-zos < %s | FileCheck %s
; REQUIRES: systemz-registered-target

; CHECK: @@CM_0: * @void_test
; CHECK: * XPLINK Routine Layout Entry
; CHECK: .long   12779717 * Eyecatcher 0x00C300C500C500
; CHECK: .short  197
; CHECK: .byte   0
; CHECK: .byte   241 * Mark Type C'1'
; CHECK: .long   128 * DSA Size 0x80
; CHECK: * Entry Flags
; CHECK: *   Bit 2: 0 = Does not use alloca
define void @void_test() {
entry:
  ret void
}

; RUN: llc -mtriple s390x-ibm-zos < %s | FileCheck %s
; REQUIRES: systemz-registered-target

; CHECK: L#EPM_void_test_0: * @void_test
; CHECK: * XPLINK Routine Layout Entry
; CHECK: .long   12779717 * Eyecatcher 0x00C300C500C500
; CHECK: .short  197
; CHECK: .byte   0
; CHECK: .byte   241 * Mark Type C'1'
; CHECK: .long   8 * DSA Size 0x0
; CHECK: * Entry Flags
; CHECK: *   Bit 1: 1 = Leaf function
; CHECK: *   Bit 2: 0 = Does not use alloca
; CHECK: L#func_end0:
; CHECK: .section        ".ppa1"
; CHECK: L#PPA1_void_test_0:                     * PPA1
; CHECK:        .byte   2                               * Version
; CHECK:        .byte   206                             * LE Signature X'CE'
; CHECK:        .short  0                               * Saved GPR Mask
; CHECK:        .byte   128                             * PPA1 Flags 1
; CHECK:                                        *   Bit 0: 1 = 64-bit DSA
; CHECK:        .byte   128                             * PPA1 Flags 2
; CHECK:                                        *   Bit 0: 1 = External procedure
; CHECK:                                        *   Bit 3: 0 = STACKPROTECT is not enabled
; CHECK:         .byte   0                               * PPA1 Flags 3
; CHECK:        .byte   129                             * PPA1 Flags 4
; CHECK:        .short  0                               * Length/4 of Parms
; CHECK:        .long   L#func_end0-L#EPM_void_test_0   * Length of Code
; CHECK:        .long   L#EPM_void_test_0-L#PPA1_void_test_0
; CHECK:        .section        ".text"
; CHECK:                                        * -- End function
define void @void_test() {
entry:
  ret void
}

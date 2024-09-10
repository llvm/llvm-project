; RUN: llc -mtriple s390x-ibm-zos < %s | FileCheck %s

; CHECK:      L#EPM_void_test_0:   * @void_test
; CHECK-NEXT:                      * XPLINK Routine Layout Entry
; CHECK-NEXT: .long   12779717     * Eyecatcher 0x00C300C500C500
; CHECK-NEXT: .short  197
; CHECK-NEXT: .byte   0
; CHECK-NEXT: .byte   241          * Mark Type C'1'
; CHECK-NEXT: .long	L#PPA1_void_test_0-L#EPM_void_test_0 * Offset to PPA1
; CHECK-NEXT: .long   8            * DSA Size 0x0
; CHECK-NEXT: * Entry Flags
; CHECK-NEXT: *   Bit 1: 1 = Leaf function
; CHECK-NEXT: *   Bit 2: 0 = Does not use alloca
;
; CHECK: L#func_end0:
; CHECK-NEXT: .section        ".text"
; CHECK-NEXT: .subsection	    2
; CHECK-NEXT: L#PPA1_void_test_0:                     * PPA1
; CHECK-NEXT:        .byte   2                               * Version
; CHECK-NEXT:        .byte   206                             * LE Signature X'CE'
; CHECK-NEXT:        .short  0                               * Saved GPR Mask
; CHECK-NEXT:        .long	 L#PPA2-L#PPA1_void_test_0       * Offset to PPA2
; CHECK-NEXT:        .byte   128                             * PPA1 Flags 1
; CHECK-NEXT:                                        *   Bit 0: 1 = 64-bit DSA
; CHECK-NEXT:        .byte   128                             * PPA1 Flags 2
; CHECK-NEXT:                                        *   Bit 0: 1 = External procedure
; CHECK-NEXT:                                        *   Bit 3: 0 = STACKPROTECT is not enabled
; CHECK-NEXT:        .byte   0                               * PPA1 Flags 3
; CHECK-NEXT:        .byte   129                             * PPA1 Flags 4
; CHECK-NEXT:                                        *   Bit 7: 1 = Name Length and Name
; CHECK-NEXT:        .short  0                               * Length/4 of Parms
; CHECK-NEXT:        .long   L#func_end0-L#EPM_void_test_0   * Length of Code
; CHECK-NEXT:        .short	 9                               * Length of Name
; CHECK-NEXT:        .ascii	 "\245\226\211\204m\243\205\242\243" * Name of Function
; CHECK-NEXT:        .space	 1
; CHECK-NEXT:        .long   L#EPM_void_test_0-L#PPA1_void_test_0
; CHECK-NEXT:        .section        ".text"
; CHECK-NEXT:                                        * -- End function
define void @void_test() {
entry:
  ret void
}

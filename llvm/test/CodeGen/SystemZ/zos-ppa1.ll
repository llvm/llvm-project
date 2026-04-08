; RUN: llc -mtriple s390x-ibm-zos < %s | FileCheck %s

; CHECK: * XPLINK Routine Layout Entry
; CHECK: L#EPM_void_test_0 DS 0H
; CHECK: * Eyecatcher 0x00C300C500C500
; CHECK:  DC XL7'00C300C500C500'
; CHECK: * Mark Type C'1'
; CHECK:  DC XL1'F1'
; CHECK: * Offset to PPA1
; CHECK:  DC AD(L#PPA1_void_test_0-L#EPM_void_test_0)
; CHECK: * DSA Size 0x0
; CHECK: * Entry Flags
; CHECK: *   Bit 1: 1 = Leaf function
; CHECK: *   Bit 2: 0 = Does not use alloca
; CHECK:  DC XL4'00000008'
; CHECK:  ENTRY void_test
; CHECK: L#func_end0 DS 0H
; CHECK: stdin#C CSECT
; CHECK: C_CODE64 CATTR
; CHECK: * PPA1
; CHECK: L#PPA1_void_test_0 DS 0H
; CHECK: * Version
; CHECK:  DC XL1'02'
; CHECK: * LE Signature X'CE'
; CHECK:  DC XL1'CE'
; CHECK: * Saved GPR Mask
; CHECK:  DC XL2'0000'
; CHECK: * Offset to PPA2
; CHECK:  DC AD(L#PPA2-L#PPA1_void_test_0)
; CHECK: * PPA1 Flags 1
; CHECK: *   Bit 0: 1 = 64-bit DSA
; CHECK:  DC XL1'80'
; CHECK: * PPA1 Flags 2
; CHECK: *   Bit 0: 1 = External procedure
; CHECK: *   Bit 3: 0 = STACKPROTECT is not enabled
; CHECK:  DC XL1'80'
; CHECK: * PPA1 Flags 3
; CHECK:  DC XL1'00'
; CHECK: * PPA1 Flags 4
; CHECK: *   Bit 7: 1 = Name Length and Name
; CHECK:  DC XL1'81'
; CHECK: * Length/4 of Parms
; CHECK:  DC XL2'0000'
; CHECK: * Length of Code
; CHECK:  DC AD(L#func_end0-L#EPM_void_test_0)
; CHECK: * Length of Name
; CHECK:  DC XL2'0009'
; CHECK: * Name of Function
; CHECK:  DC XL9'A59689846DA385A2A3'
; CHECK: DC AD(L#EPM_void_test_0-L#PPA1_void_test_0)
define void @void_test() {
entry:
  ret void
}

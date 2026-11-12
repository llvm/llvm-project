; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr7 < %s | \
; RUN:   FileCheck %s --check-prefixes=CHECK,CHECK32

; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr7 < %s | \
; RUN:   FileCheck %s --check-prefixes=CHECK,CHECK64


@NOT_PI = constant double 3.000000e+00, align 8

define double @PIDouble() #0 {
  entry:
    ret double 3.141590e+00
}

define float @PIfloat() #0 {
  entry:
    ret float 0x400921FA00000000
}

; CHECK:         .csect .rodata.8[RO],3
; CHECK-NEXT:    .align  3
; CHECK-NEXT:  L..CPI0_0:
; CHECK32-NEXT:  .vbyte  4, 1074340345
; CHECK32-NEXT:  .vbyte  4, 4028335726
; CHECK64-NEXT:  .vbyte  8, 0x400921f9f01b866e


; CHECK:         .csect .rodata[RO],2
; CHECK-NEXT:    .align  2
; CHECK-NEXT:  L..CPI1_0:
; CHECK-NEXT:    .vbyte  4, 0x40490fd0

; CHECK:         .csect NOT_PI[RO],3

; RUN: llc -filetype=obj -mtriple=aarch64 -mattr=+execute-only %s -o %t.o
; RUN: llvm-readobj -S %t.o | FileCheck %s

; CHECK:         Name: .text
; CHECK-NEXT:    Type: SHT_PROGBITS
; CHECK-NEXT:    Flags [
; CHECK-NEXT:      SHF_AARCH64_PURECODE
; CHECK-NEXT:      SHF_ALLOC
; CHECK-NEXT:      SHF_EXECINSTR
; CHECK-NEXT:    ]
; CHECK-NEXT:    Address:
; CHECK-NEXT:    Offset:
; CHECK-NEXT:    Size: 0

# RUN: llvm-mc -filetype=obj -triple=msp430 %s | llvm-readobj -t - | FileCheck %s

foo:
  .refsym __hook

; CHECK:       Symbol {
; CHECK:         Name: __hook (30)
; CHECK-NEXT:    Value: 0x0
; CHECK-NEXT:    Size: 0
; CHECK-NEXT:    Binding: Global (0x1)
; CHECK-NEXT:    Type: None (0x0)
; CHECK-NEXT:    Other: 0
; CHECK-NEXT:    Section: Undefined (0x0)
; CHECK-NEXT:  }

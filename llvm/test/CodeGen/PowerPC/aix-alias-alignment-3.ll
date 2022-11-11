;; TODO: The alias offset doesn't refer to any sub-element.
; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -data-sections=false < %s | FileCheck %s

;; FIXME: The alias is not emitted in the correct location.

@ConstVector = global <2 x i64> <i64 1, i64 2>
@var = alias i64, getelementptr inbounds (i8, ptr @ConstVector, i32 1)

; CHECK:              .csect .data[RW],4
; CHECK-NEXT:         .globl  ConstVector                     # @ConstVector
; CHECK-NEXT:         .globl  var
; CHECK-NEXT:         .align  4
; CHECK-NEXT: ConstVector:
; CHECK-NEXT:         .vbyte  4, 0                            # 0x1
; CHECK-NEXT:         .vbyte  4, 1
; CHECK-NEXT:         .vbyte  4, 0                            # 0x2
; CHECK-NEXT:         .vbyte  4, 2
; CHECK-NEXT: var:

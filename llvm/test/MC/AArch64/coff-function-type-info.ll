; RUN: llc -mtriple arm64-windows -filetype asm -o - %s \
; RUN:    | FileCheck %s -check-prefix CHECK-ASM

; RUN: llc -mtriple arm64-windows -filetype obj -o - %s \
; RUN:    | llvm-readobj --symbols | FileCheck %s -check-prefix CHECK-OBJECT

define arm_aapcs_vfpcc void @external() {
entry:
  ret void
}

; CHECK-ASM: .def external
; CHECK-ASM:   .scl 2
; CHECK-ASM:   .type 32
; CHECK-ASM: .endef
; CHECK-ASM: .globl external

define internal arm_aapcs_vfpcc void @internal() {
entry:
  ret void
}

; CHECK-ASM: .def internal
; CHECK-ASM:    .scl 3
; CHECK-ASM:    .type 32
; CHECK-ASM: .endef
; CHECK-ASM-NOT: .globl internal

; CHECK-OBJECT: Symbol {
; CHECK-OBJECT:   Name: external
; CHECK-OBJECT:   Section: .text
; CHECK-OBJECT:   BaseType: Null
; CHECK-OBJECT:   ComplexType: Function
; CHECK-OBJECT:   StorageClass: External
; CHECK-OBJECT:   AuxSymbolCount: 0
; CHECK-OBJECT: }
; CHECK-OBJECT: Symbol {
; CHECK-OBJECT:   Name: internal
; CHECK-OBJECT:   Section: .text
; CHECK-OBJECT:   BaseType: Null
; CHECK-OBJECT:   ComplexType: Function
; CHECK-OBJECT:   StorageClass: Static
; CHECK-OBJECT:   AuxSymbolCount: 0
; CHECK-OBJECT: }


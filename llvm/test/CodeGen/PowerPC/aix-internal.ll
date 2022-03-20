; RUN: llc -mtriple powerpc-ibm-aix -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec \
; RUN: -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --syms %t.o | FileCheck %s

;; FIXME: currently only fileHeader and sectionHeaders are supported in XCOFF64.

define internal i32 @foo() {
  ret i32 1
}

; CHECK:  Symbol {
; CHECK:       Name: .foo
; CHECK-NEXT:  Value (RelocatableAddress):
; CHECK-NEXT:  Section: .text
; CHECK-NEXT:  Type: 0x0
; CHECK-NEXT:  StorageClass: C_HIDEXT (0x6B)

; CHECK:  Symbol {
; CHECK-NEXT: Index: [[#INDX:]]
; CHECK-NEXT: Name: foo
; CHECK-NEXT: Value (RelocatableAddress):
; CHECK-NEXT: Section: .data
; CHECK-NEXT: Type: 0x0
; CHECK-NEXT: StorageClass: C_HIDEXT (0x6B)
; CHECK-NEXT: NumberOfAuxEntries: 1
; CHECK-NEXT: CSECT Auxiliary Entry {
; CHECK-NEXT:  Index: [[#INDX+1]]
; CHECK-NEXT:  SectionLen: 12
; CHECK-NEXT:  ParameterHashIndex:
; CHECK-NEXT:  TypeChkSectNum:
; CHECK-NEXT:  SymbolAlignmentLog2:
; CHECK-NEXT:  SymbolType: XTY_SD (0x1)
; CHECK-NEXT:  StorageMappingClass: XMC_DS (0xA)

; Make sure no label is emitted.
; CHECK-NOT: Name: foo

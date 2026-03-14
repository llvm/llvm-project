; Test that the -mcpu= option sets the correct ELF build attributes.

; RUN: llc -mtriple=msp430 -filetype=obj < %s \
; RUN:   | llvm-readelf -A - | FileCheck %s --check-prefixes COMMON,MSP430,SMALL
; RUN: llc -mtriple=msp430 -mcpu=generic -filetype=obj < %s \
; RUN:   | llvm-readelf -A - | FileCheck %s --check-prefixes COMMON,MSP430,SMALL
; RUN: llc -mtriple=msp430 -mcpu=msp430 -filetype=obj < %s \
; RUN:   | llvm-readelf -A - | FileCheck %s --check-prefixes COMMON,MSP430,SMALL
; RUN: llc -mtriple=msp430 -mcpu=msp430x -filetype=obj < %s \
; RUN:   | llvm-readelf -A - | FileCheck %s --check-prefixes COMMON,MSP430X,SMALL

; COMMON: BuildAttributes {
; COMMON: FormatVersion: 0x41
; COMMON:   SectionLength: 22
; COMMON:   Vendor: mspabi
; COMMON:   Tag: Tag_File (0x1)
; COMMON:   Size: 11

; MSP430:      Tag: 4
; MSP430-NEXT: Value: 1
; MSP430-NEXT: TagName: ISA
; MSP430-NEXT: Description: MSP430

; MSP430X:      Tag: 4
; MSP430X-NEXT: Value: 2
; MSP430X-NEXT: TagName: ISA
; MSP430X-NEXT: Description: MSP430X

; SMALL:      Tag: 6
; SMALL-NEXT: Value: 1
; SMALL-NEXT: TagName: Code_Model
; SMALL-NEXT: Description: Small

; SMALL:      Tag: 8
; SMALL-NEXT: Value: 1
; SMALL-NEXT: TagName: Data_Model
; SMALL-NEXT: Description: Small

define void @foo() {
  ret void
}

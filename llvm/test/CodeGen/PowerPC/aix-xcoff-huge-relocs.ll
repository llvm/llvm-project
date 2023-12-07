;; This test takes a very long time
; REQUIRES: expensive_checks

;; This test generates 65535 relocation entries in a single section,
;; which would trigger an overflow section to be generated in 32-bit mode.
; RUN: grep -v RUN: %s | \
; RUN:   sed > %t.overflow.ll 's/SIZE/65535/;s/MACRO/#/;s/#/################/g;s/#/################/g;s/#/################/g;s/#/################/g;s/#/#_/g;s/_#_\([^#]\)/\1/;s/_/, /g;s/#/ptr @c/g;'
; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff \
; RUN:     -mcpu=pwr4 -mattr=-altivec -filetype=obj -o %t.overflow.o %t.overflow.ll
; RUN: llvm-readobj --section-headers %t.overflow.o | FileCheck --check-prefix=OVERFLOW %s

;; This test generates 65534 relocation entries, an overflow section should
;; not be generated.
; RUN: grep -v RUN: %s | \
; RUN:   sed >%t.ll 's/SIZE/65534/;s/MACRO/#/;s/#/################/g;s/#/################/g;s/#/################/g;s/#/################/g;s/#/#_/g;s/_#_#_\([^#]\)/\1/;s/_/, /g;s/#/ptr @c/g;'
; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff \
; RUN:     -mcpu=pwr4 -mattr=-altivec -filetype=obj -o %t.o %t.ll
; RUN: llvm-readobj --section-headers %t.o | FileCheck --check-prefix=XCOFF32 %s

;; An XCOFF64 file may not contain an overflow section header.
; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff \
; RUN:     -mcpu=pwr4 -mattr=-altivec -filetype=obj -o %t64.o %t.overflow.ll
; RUN: llvm-readobj --section-headers %t64.o | FileCheck --check-prefix=XCOFF64 %s

@c = external global i8, align 1
@arr = global [SIZE x ptr] [MACRO], align 8

; OVERFLOW:      Sections [
; OVERFLOW-NEXT:   Section {
; OVERFLOW-NEXT:     Index: 1
; OVERFLOW-NEXT:     Name: .text
; OVERFLOW-NEXT:     PhysicalAddress: 0x0
; OVERFLOW-NEXT:     VirtualAddress: 0x0
; OVERFLOW-NEXT:     Size: 0x0
; OVERFLOW-NEXT:     RawDataOffset: 0x8C
; OVERFLOW-NEXT:     RelocationPointer: 0x0
; OVERFLOW-NEXT:     LineNumberPointer: 0x0
; OVERFLOW-NEXT:     NumberOfRelocations: 0
; OVERFLOW-NEXT:     NumberOfLineNumbers: 0
; OVERFLOW-NEXT:     Type: STYP_TEXT (0x20)
; OVERFLOW-NEXT:   }
; OVERFLOW-NEXT:   Section {
; OVERFLOW-NEXT:     Index: 2
; OVERFLOW-NEXT:     Name: .data
; OVERFLOW-NEXT:     PhysicalAddress: 0x0
; OVERFLOW-NEXT:     VirtualAddress: 0x0
; OVERFLOW-NEXT:     Size: 0x3FFFC
; OVERFLOW-NEXT:     RawDataOffset: 0x8C
; OVERFLOW-NEXT:     RelocationPointer: 0x40088
; OVERFLOW-NEXT:     LineNumberPointer: 0x0
; OVERFLOW-NEXT:     NumberOfRelocations: 65535
; OVERFLOW-NEXT:     NumberOfLineNumbers: 65535
; OVERFLOW-NEXT:     Type: STYP_DATA (0x40)
; OVERFLOW-NEXT:   }
; OVERFLOW-NEXT:   Section {
; OVERFLOW-NEXT:     Index: 3
; OVERFLOW-NEXT:     Name: .ovrflo
; OVERFLOW-NEXT:     NumberOfRelocations: 65535
; OVERFLOW-NEXT:     NumberOfLineNumbers: 0
; OVERFLOW-NEXT:     Size: 0x0
; OVERFLOW-NEXT:     RawDataOffset: 0x0
; OVERFLOW-NEXT:     RelocationPointer: 0x40088
; OVERFLOW-NEXT:     LineNumberPointer: 0x0
; OVERFLOW-NEXT:     IndexOfSectionOverflowed: 2
; OVERFLOW-NEXT:     IndexOfSectionOverflowed: 2
; OVERFLOW-NEXT:     Type: STYP_OVRFLO (0x8000)
; OVERFLOW-NEXT:   }
; OVERFLOW-NEXT: ]

; XCOFF32:       Section {
; XCOFF32:         Name: .data
; XCOFF32-NEXT:    PhysicalAddress: 0x0
; XCOFF32-NEXT:    VirtualAddress: 0x0
; XCOFF32-NEXT:    Size: 0x3FFF8
; XCOFF32-NEXT:    RawDataOffset: 0x64
; XCOFF32-NEXT:    RelocationPointer: 0x4005C
; XCOFF32-NEXT:    LineNumberPointer: 0x0
; XCOFF32-NEXT:    NumberOfRelocations: 65534
; XCOFF32-NEXT:    NumberOfLineNumbers: 0
; XCOFF32-NEXT:    Type: STYP_DATA (0x40)
; XCOFF32-NEXT:  }

; XCOFF64:      Section {
; XCOFF64:        Name: .data
; XCOFF64-NEXT:   PhysicalAddress: 0x0
; XCOFF64-NEXT:   VirtualAddress: 0x0
; XCOFF64-NEXT:   Size: 0x7FFF8
; XCOFF64-NEXT:   RawDataOffset: 0xA8
; XCOFF64-NEXT:   RelocationPointer: 0x800A0
; XCOFF64-NEXT:   LineNumberPointer: 0x0
; XCOFF64-NEXT:   NumberOfRelocations: 65535
; XCOFF64-NEXT:   NumberOfLineNumbers: 0
; XCOFF64-NEXT:   Type: STYP_DATA (0x40)
; XCOFF64-NEXT: }

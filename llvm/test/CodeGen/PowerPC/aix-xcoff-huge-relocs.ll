;; This test takes a very long time
; REQUIRES: expensive_checks

;; This test generates 65535 relocation entries in a single section,
;; which would trigger an overflow section to be generated in 32-bit mode.
;; Since overflow section is not supported yet, we will emit an error instead of
;; generating an invalid binary for now.
; RUN: grep -v RUN: %s | \
; RUN:   sed >%t.overflow.ll 's/SIZE/65535/;s/MACRO/#/;s/#/################/g;s/#/################/g;s/#/################/g;s/#/################/g;s/#/#_/g;s/_#_\([^#]\)/\1/;s/_/, /g;s/#/ptr @c/g;'
; RUN: not --crash llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff \
; RUN:                 -mcpu=pwr4 -mattr=-altivec -filetype=obj -o %t.o %t.overflow.ll 2>&1 | \
; RUN:   FileCheck --check-prefix=OVERFLOW %s
; OVERFLOW: LLVM ERROR: relocation entries overflowed; overflow section is not implemented yet

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

; XCOFF32-NOT:     Name: .ovrflo
; XCOFF32-NOT:     Type: STYP_OVRFLO
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
; XCOFF32-NOT:     Name: .ovrflo
; XCOFF32-NOT:     Type: STYP_OVRFLO

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

; The purpose of this test is to verify that various module level assembly
; constructs work.

; RUN: llc -filetype=obj -mtriple i686-pc-win32 %s -o - | llvm-readobj -S --sd - | FileCheck %s
; RUN: llc -filetype=obj -mtriple x86_64-pc-win32 %s -o - | llvm-readobj -S --sd - | FileCheck %s

module asm ".text"
module asm "_foo:"
module asm "  ret"

; CHECK:            Name:                      .text
; CHECK-NEXT:       VirtualSize:               0
; CHECK-NEXT:       VirtualAddress:            0
; CHECK-NEXT:       RawDataSize:               {{[0-9]+}}
; CHECK-NEXT:       PointerToRawData:          0x{{[0-9A-F]+}}
; CHECK-NEXT:       PointerToRelocations:      0x{{[0-9A-F]+}}
; CHECK-NEXT:       PointerToLineNumbers:      0x0
; CHECK-NEXT:       RelocationCount:           0
; CHECK-NEXT:       LineNumberCount:           0
; CHECK-NEXT:       Characteristics [ (0x60300020)
; CHECK-NEXT:         IMAGE_SCN_ALIGN_4BYTES
; CHECK-NEXT:         IMAGE_SCN_CNT_CODE
; CHECK-NEXT:         IMAGE_SCN_MEM_EXECUTE
; CHECK-NEXT:         IMAGE_SCN_MEM_READ
; CHECK-NEXT:       ]
; CHECK-NEXT:       SectionData (
; CHECK-NEXT:         0000: C3
; CHECK-NEXT:       )

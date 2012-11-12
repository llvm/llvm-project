; The purpose of this test is to verify that various module level assembly
; constructs work.

; RUN: llc -filetype=obj -mtriple i686-pc-win32 %s -o - | coff-dump.py | FileCheck %s
; RUN: llc -filetype=obj -mtriple x86_64-pc-win32 %s -o - | coff-dump.py | FileCheck %s

module asm ".text"
module asm "_foo:"
module asm "  ret"

; CHECK:            Name                     = .text
; CHECK-NEXT:       VirtualSize              = 0
; CHECK-NEXT:       VirtualAddress           = 0
; CHECK-NEXT:       SizeOfRawData            = {{[0-9]+}}
; CHECK-NEXT:       PointerToRawData         = 0x{{[0-9A-F]+}}
; CHECK-NEXT:       PointerToRelocations     = 0x{{[0-9A-F]+}}
; CHECK-NEXT:       PointerToLineNumbers     = 0x0
; CHECK-NEXT:       NumberOfRelocations      = 0
; CHECK-NEXT:       NumberOfLineNumbers      = 0
; CHECK-NEXT:       Charateristics           = 0x60300020
; CHECK-NEXT:         IMAGE_SCN_CNT_CODE
; CHECK-NEXT:         IMAGE_SCN_ALIGN_4BYTES
; CHECK-NEXT:         IMAGE_SCN_MEM_EXECUTE
; CHECK-NEXT:         IMAGE_SCN_MEM_READ
; CHECK-NEXT:       SectionData              =
; CHECK-NEXT:         C3

// RUN: llvm-mc -filetype=obj -triple i686-pc-win32 %s -o - | coff-dump.py | FileCheck %s

// Test that we get optimal nops in text
    .text
f0:
    .long 0
    .align  8, 0x90
    .long 0
    .align  8

// But not in another section
    .data
    .long 0
    .align  8, 0x90
    .long 0
    .align  8

//CHECK:         Name                     = .text
//CHECK-NEXT:    VirtualSize
//CHECK-NEXT:    VirtualAddress
//CHECK-NEXT:    SizeOfRawData            = 16
//CHECK-NEXT:    PointerToRawData
//CHECK-NEXT:    PointerToRelocations
//CHECK-NEXT:    PointerToLineNumbers
//CHECK-NEXT:    NumberOfRelocations
//CHECK-NEXT:    NumberOfLineNumbers
//CHECK-NEXT:    Charateristics           = 0x60400020
//CHECK-NEXT:        IMAGE_SCN_CNT_CODE
//CHECK-NEXT:        IMAGE_SCN_ALIGN_8BYTES
//CHECK-NEXT:        IMAGE_SCN_MEM_EXECUTE
//CHECK-NEXT:        IMAGE_SCN_MEM_READ
//CHECK-NEXT:      SectionData              =
//CHECK-NEXT:        00 00 00 00 0F 1F 40 00 - 00 00 00 00 0F 1F 40 00

//CHECK:         Name                     = .data
//CHECK-NEXT:      VirtualSize
//CHECK-NEXT:      VirtualAddress
//CHECK-NEXT:      SizeOfRawData            = 16
//CHECK-NEXT:      PointerToRawData
//CHECK-NEXT:      PointerToRelocations
//CHECK-NEXT:      PointerToLineNumbers
//CHECK-NEXT:      NumberOfRelocations
//CHECK-NEXT:      NumberOfLineNumbers
//CHECK-NEXT:      Charateristics           = 0xC0400040
//CHECK-NEXT:        IMAGE_SCN_CNT_INITIALIZED_DATA
//CHECK-NEXT:        IMAGE_SCN_ALIGN_8BYTES
//CHECK-NEXT:        IMAGE_SCN_MEM_READ
//CHECK-NEXT:        IMAGE_SCN_MEM_WRITE
//CHECK-NEXT:      SectionData              =
//CHECK-NEXT:        00 00 00 00 90 90 90 90 - 00 00 00 00 00 00 00 00

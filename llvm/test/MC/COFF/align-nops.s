// RUN: llvm-mc -filetype=obj -triple i686-pc-win32 -mcpu=pentiumpro %s | llvm-readobj -S --sd | FileCheck %s

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

//CHECK:          Name: .text
//CHECK-NEXT:     VirtualSize
//CHECK-NEXT:     VirtualAddress
//CHECK-NEXT:     RawDataSize: 16
//CHECK-NEXT:     PointerToRawData
//CHECK-NEXT:     PointerToRelocations
//CHECK-NEXT:     PointerToLineNumbers
//CHECK-NEXT:     RelocationCount
//CHECK-NEXT:     LineNumberCount
//CHECK-NEXT:     Characteristics [ (0x60400020)
//CHECK-NEXT:        IMAGE_SCN_ALIGN_8BYTES
//CHECK-NEXT:        IMAGE_SCN_CNT_CODE
//CHECK-NEXT:        IMAGE_SCN_MEM_EXECUTE
//CHECK-NEXT:        IMAGE_SCN_MEM_READ
//CHECK-NEXT:     ]
//CHECK-NEXT:     SectionData (
//CHECK-NEXT:       0000: 00000000 0F1F4000 00000000 0F1F4000
//CHECK-NEXT:     )

//CHECK:          Name: .data
//CHECK-NEXT:     VirtualSize:
//CHECK-NEXT:     VirtualAddress:
//CHECK-NEXT:     RawDataSize: 16
//CHECK-NEXT:     PointerToRawData:
//CHECK-NEXT:     PointerToRelocations:
//CHECK-NEXT:     PointerToLineNumbers:
//CHECK-NEXT:     RelocationCount:
//CHECK-NEXT:     LineNumberCount:
//CHECK-NEXT:     Characteristics [ (0xC0400040)
//CHECK-NEXT:       IMAGE_SCN_ALIGN_8BYTES
//CHECK-NEXT:       IMAGE_SCN_CNT_INITIALIZED_DATA
//CHECK-NEXT:       IMAGE_SCN_MEM_READ
//CHECK-NEXT:       IMAGE_SCN_MEM_WRITE
//CHECK-NEXT:     ]
//CHECK-NEXT:     SectionData (
//CHECK-NEXT:       0000: 00000000 90909090 00000000 00000000
//CHECK-NEXT:     )

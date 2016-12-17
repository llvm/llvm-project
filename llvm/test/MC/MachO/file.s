// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | llvm-readobj -s -section-data | FileCheck %s

        .file	1 "dir/foo"
        nop

// CHECK:       Section {
// CHECK:         Index: 1
// CHECK-NEXT:    Name: __debug_line
// CHECK-NEXT:    Segment: __DWARF
// CHECK-NEXT:    Address: 0x1
// CHECK-NEXT:    Size: 0x28
// CHECK-NEXT:    Offset: 237
// CHECK-NEXT:    Alignment: 0
// CHECK-NEXT:    RelocationOffset: 0x0
// CHECK-NEXT:    RelocationCount: 0
// CHECK-NEXT:    Type: 0x0
// CHECK-NEXT:    Attributes [ (0x20000)
// CHECK-NEXT:      Debug (0x20000)
// CHECK-NEXT:    ]
// CHECK-NEXT:    Reserved1: 0x0
// CHECK-NEXT:    Reserved2: 0x0
// CHECK-NEXT:    SectionData (
// CHECK-NEXT:      0000: 24000000 02001E00 00000101 FB0E0D00
// CHECK-NEXT:      0010: 01010101 00000001 00000164 69720000
// CHECK-NEXT:      0020: 666F6F00 01000000
// CHECK-NEXT:    )
// CHECK-NEXT:  }

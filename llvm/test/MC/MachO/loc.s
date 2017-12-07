// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | llvm-readobj -r -s -sd | FileCheck %s

        .file	1 "foo"
	.loc	1 64 0
        nop

// CHECK: Section {
// CHECK:     Index: 1
// CHECK:     Name: __debug_line (5F 5F 64 65 62 75 67 5F 6C 69 6E 65 00 00 00 00)
// CHECK:     Segment: __DWARF (5F 5F 44 57 41 52 46 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x1
// CHECK:     Size: 0x33
// CHECK:     Offset: 237
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x120
// CHECK:     RelocationCount: 1
// CHECK:     Type: 0x0
// CHECK:     Attributes [ (0x20000)
// CHECK:       Debug (0x20000)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:       0000: 2F000000 02001A00 00000101 FB0E0D00  |/...............|
// CHECK:       0010: 01010101 00000001 00000100 666F6F00  |............foo.|
// CHECK:       0020: 00000000 00050200 00000003 3F010201  |............?...|
// CHECK:       0030: 000101                               |...|
// CHECK:     )
// CHECK:   }
// CHECK: ]
// CHECK: Relocations [
// CHECK:   Section __debug_line {
// CHECK:     0x27 0 2 0 GENERIC_RELOC_VANILLA 0 __text
// CHECK:   }
// CHECK: ]

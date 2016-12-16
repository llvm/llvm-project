// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | llvm-readobj -r -s -sd | FileCheck %s

        .file	1 "foo"
	.loc	1 64 0
        nop

// CHECK: Section {
// CHECK:     Index: 1
// CHECK:     Name: __debug_line (5F 5F 64 65 62 75 67 5F 6C 69 6E 65 00 00 00 00)
// CHECK:     Segment: __DWARF (5F 5F 44 57 41 52 46 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x1
// CHECK:     Size: 0x34
// CHECK:     Offset: 237
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x124
// CHECK:     RelocationCount: 1
// CHECK:     Type: 0x0
// CHECK:     Attributes [ (0x20000)
// CHECK:       Debug (0x20000)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:       0000: 30000000 04001B00 00000101 01FB0E0D  |0...............|
// CHECK:       0010: 00010101 01000000 01000001 00666F6F  |.............foo|
// CHECK:       0020: 00000000 00000502 00000000 033F0102  |.............?..|
// CHECK:       0030: 01000101   
// CHECK:     )
// CHECK:   }
// CHECK: ]
// CHECK: Relocations [
// CHECK:   Section __debug_line {
// CHECK:     0x28 0 2 0 GENERIC_RELOC_VANILLA 0 __text
// CHECK:   }
// CHECK: ]

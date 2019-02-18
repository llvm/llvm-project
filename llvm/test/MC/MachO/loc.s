// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | llvm-readobj -r -s -sd | FileCheck %s
        .file   2 "foo"
        .file   1 "bar"
        .loc    2 64 0
        nop

// CHECK: Section {
// CHECK:     Index: 1
// CHECK:     Name: __debug_line (5F 5F 64 65 62 75 67 5F 6C 69 6E 65 00 00 00 00)
// CHECK:     Segment: __DWARF (5F 5F 44 57 41 52 46 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x1
// CHECK:     Size: 0x3D
// CHECK:     Offset: 237
// CHECK:     Alignment: 0
// CHECK:     RelocationOffset: 0x12C
// CHECK:     RelocationCount: 1
// CHECK:     Type: 0x0
// CHECK:     Attributes [ (0x20000)
// CHECK:       Debug (0x20000)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:       000: 39000000 04002200 00000101 01FB0E0D  |9.....".........|
// CHECK:       010: 00010101 01000000 01000001 00626172  |.............bar|
// CHECK:       020: 00000000 666F6F00 00000000 04020005  |....foo.........|
// CHECK:       030: 02000000 00033F01 02010001 01        |......?......|
// CHECK:     )
// CHECK:   }
// CHECK: ]
// CHECK: Relocations [
// CHECK:   Section __debug_line {
// CHECK:     0x31 0 2 0 GENERIC_RELOC_VANILLA 0 __text
// CHECK:   }
// CHECK: ]

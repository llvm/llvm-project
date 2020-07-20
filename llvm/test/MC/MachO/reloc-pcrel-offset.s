// RUN: llvm-mc -n -triple i386-apple-darwin9 %s -filetype=obj -o - | llvm-readobj -r -S --sd - | FileCheck %s

        .data
        .long 0

        .text
_a:
_b:
        call _a

        .subsections_via_symbols

// CHECK: Section {
// CHECK:   Index: 0
// CHECK:   Name: __data (5F 5F 64 61 74 61 00 00 00 00 00 00 00 00 00 00)
// CHECK:   Segment: __DATA (5F 5F 44 41 54 41 00 00 00 00 00 00 00 00 00 00)
// CHECK:   Address: 0x0
// CHECK:   Size: 0x4
// CHECK:   Offset: 340
// CHECK:   Alignment: 0
// CHECK:   RelocationOffset: 0x0
// CHECK:   RelocationCount: 0
// CHECK:   Type: Regular (0x0)
// CHECK:   Attributes [ (0x0)
// CHECK:   ]
// CHECK:   Reserved1: 0x0
// CHECK:   Reserved2: 0x0
// CHECK:   SectionData (
// CHECK:     0000: 00000000                             |....|
// CHECK:   )
// CHECK: }
// CHECK: Relocations [
// CHECK:   Section __text {
// CHECK:     0x1 1 2 0 GENERIC_RELOC_VANILLA 0 __text
// CHECK:   }
// CHECK: ]

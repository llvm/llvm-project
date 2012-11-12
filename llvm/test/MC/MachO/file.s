// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | macho-dump --dump-section-data | FileCheck %s

        .file	1 "dir/foo"
        nop

// CHECK:         ('_section_data', '90')
// CHECK-NEXT:      # Section 1
// CHECK-NEXT:     (('section_name', '__debug_line\x00\x00\x00\x00')
// CHECK-NEXT:      ('segment_name', '__DWARF\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK-NEXT:      ('address', 1)
// CHECK-NEXT:      ('size', 45)
// CHECK-NEXT:      ('offset', 221)
// CHECK-NEXT:      ('alignment', 0)
// CHECK-NEXT:      ('reloc_offset', 0)
// CHECK-NEXT:      ('num_reloc', 0)
// CHECK-NEXT:      ('flags', 0x2000000)
// CHECK-NEXT:      ('reserved1', 0)
// CHECK-NEXT:      ('reserved2', 0)
// CHECK-NEXT:     ),
// CHECK-NEXT:    ('_relocations', [
// CHECK-NEXT:    ])
// CHECK-NEXT:    ('_section_data', '29000000 02001e00 00000101 fb0e0d00 01010101 00000001 00000164 69720000 666f6f00 01000000 02000001 01')

// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | macho-dump --dump-section-data | FileCheck %s

        .file	1 "foo"
	.loc	1 64 0
        nop

// CHECK:         # Section 1
// CHECK-NEXT:   (('section_name', '__debug_line\x00\x00\x00\x00')
// CHECK-NEXT:    ('segment_name', '__DWARF\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK-NEXT:    ('address', 1)
// CHECK-NEXT:    ('size', 51)
// CHECK-NEXT:    ('offset', 221)
// CHECK-NEXT:    ('alignment', 0)
// CHECK-NEXT:    ('reloc_offset', 272)
// CHECK-NEXT:    ('num_reloc', 1)
// CHECK-NEXT:    ('flags', 0x2000000)
// CHECK-NEXT:    ('reserved1', 0)
// CHECK-NEXT:    ('reserved2', 0)
// CHECK-NEXT:   ),
// CHECK-NEXT:  ('_relocations', [
// CHECK-NEXT:    # Relocation 0
// CHECK-NEXT:    (('word-0', 0x27),
// CHECK-NEXT:     ('word-1', 0x4000001)),
// CHECK-NEXT:  ])
// CHECK-NEXT:  ('_section_data', '2f000000 02001a00 00000101 fb0e0d00 01010101 00000001 00000100 666f6f00 00000000 00050200 00000003 3f010201 000101')

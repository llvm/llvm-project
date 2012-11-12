// RUN: llvm-mc -triple x86_64-apple-darwin9 %s -filetype=obj -o - | macho-dump | FileCheck %s

// This tests that when producing files for darwin9 or older we make sure
// that debug_line sections are of a minimum size to avoid the linker bug
// described in PR8715.

        .section        __DATA,__data
        .file   1 "test.c"
        .globl  _c                      ## @c
_c:
        .asciz   "hi\n"

// CHECK:      (('section_name', '__debug_line\x00\x00\x00\x00')
// CHECK-NEXT:  ('segment_name', '__DWARF\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK-NEXT:  ('address', 4)
// CHECK-NEXT:  ('size', 44)
// CHECK-NEXT:  ('offset', 452)
// CHECK-NEXT:  ('alignment', 0)
// CHECK-NEXT:  ('reloc_offset', 0)
// CHECK-NEXT:  ('num_reloc', 0)
// CHECK-NEXT:  ('flags', 0x2000000)
// CHECK-NEXT:  ('reserved1', 0)
// CHECK-NEXT:  ('reserved2', 0)
// CHECK-NEXT:  ('reserved3', 0)
// CHECK-NEXT: ),

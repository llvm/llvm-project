// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | macho-dump | FileCheck %s

// CHECK: # Relocation 0
// CHECK: (('word-0', 0xe4000045),
// CHECK:  ('word-1', 0x4)),
// CHECK: # Relocation 1
// CHECK: (('word-0', 0xe1000000),
// CHECK:  ('word-1', 0x6)),
// CHECK: # Relocation 2
// CHECK: (('word-0', 0x40),
// CHECK:  ('word-1', 0xd000003)),
// CHECK: # Relocation 3
// CHECK: (('word-0', 0x3b),
// CHECK:  ('word-1', 0xd000003)),
// CHECK: # Relocation 4
// CHECK: (('word-0', 0x36),
// CHECK:  ('word-1', 0xd000003)),
// CHECK: # Relocation 5
// CHECK: (('word-0', 0xe0000031),
// CHECK:  ('word-1', 0x4)),
// CHECK: # Relocation 6
// CHECK: (('word-0', 0xe000002c),
// CHECK:  ('word-1', 0x4)),
// CHECK: # Relocation 7
// CHECK: (('word-0', 0x27),
// CHECK:  ('word-1', 0x5000001)),
// CHECK: # Relocation 8
// CHECK: (('word-0', 0xe0000022),
// CHECK:  ('word-1', 0x2)),
// CHECK: # Relocation 9
// CHECK: (('word-0', 0xe000001d),
// CHECK:  ('word-1', 0x2)),
// CHECK: # Relocation 10
// CHECK: (('word-0', 0x18),
// CHECK:  ('word-1', 0x5000001)),
// CHECK-NEXT: ])

        xorl %eax,%eax

        .globl _a
_a:
        xorl %eax,%eax
_b:
_d:
        xorl %eax,%eax
L0:
        xorl %eax,%eax
L1:

        call L0
        call L0 - 1
        call L0 + 1
        call _a
        call _a - 1
        call _a + 1
        call _b
        call _b - 1
        call _b + 1
        call _c
        call _c - 1
        call _c + 1
//        call _a - L0
        call _b - L0

        .subsections_via_symbols

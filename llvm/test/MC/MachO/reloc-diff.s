// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | macho-dump | FileCheck %s

// CHECK: # Relocation 0
// CHECK: (('word-0', 0xa2000014),
// CHECK:  ('word-1', 0x0)),
// CHECK: # Relocation 1
// CHECK: (('word-0', 0xa1000000),
// CHECK:  ('word-1', 0x0)),
// CHECK: # Relocation 2
// CHECK: (('word-0', 0xa4000010),
// CHECK:  ('word-1', 0x0)),
// CHECK: # Relocation 3
// CHECK: (('word-0', 0xa1000000),
// CHECK:  ('word-1', 0x0)),
// CHECK: # Relocation 4
// CHECK: (('word-0', 0xa400000c),
// CHECK:  ('word-1', 0x0)),
// CHECK: # Relocation 5
// CHECK: (('word-0', 0xa1000000),
// CHECK:  ('word-1', 0x0)),
// CHECK: # Relocation 6
// CHECK: (('word-0', 0xa4000008),
// CHECK:  ('word-1', 0x0)),
// CHECK: # Relocation 7
// CHECK: (('word-0', 0xa1000000),
// CHECK:  ('word-1', 0x0)),
// CHECK: # Relocation 8
// CHECK: (('word-0', 0xa4000004),
// CHECK:  ('word-1', 0x0)),
// CHECK: # Relocation 9
// CHECK: (('word-0', 0xa1000000),
// CHECK:  ('word-1', 0x0)),
// CHECK: # Relocation 10
// CHECK: (('word-0', 0xa2000000),
// CHECK:  ('word-1', 0x0)),
// CHECK: # Relocation 11
// CHECK: (('word-0', 0xa1000000),
// CHECK:  ('word-1', 0x0)),
// CHECK-NEXT: ])

_local_def:
        .globl _external_def
_external_def:
Ltemp:
        ret
        
        .data
        .long _external_def - _local_def
        .long Ltemp - _local_def

        .long _local_def - _external_def
        .long Ltemp - _external_def

        .long _local_def - Ltemp
        .long _external_def - Ltemp

// RUN: llvm-mc -triple x86_64-apple-darwin9 %s -filetype=obj -o - | macho-dump --dump-section-data | FileCheck %s

// Test case for rdar://10743265

// This tests that this expression does not cause a crash and produces two
// relocation entries:
// Relocation information (__TEXT,__text) 2 entries
// address  pcrel length extern type    scattered symbolnum/value
// 00000000 False long   True   SUB     False     _base
// 00000000 False long   True   UNSIGND False     _start_ap_2

_base = .

.long (0x2000) + _start_ap_2 - _base 
.word 0

_start_ap_2:
        cli

// CHECK:   ('_relocations', [
// CHECK:     # Relocation 0
// CHECK:     (('word-0', 0x0),
// CHECK:      ('word-1', 0x5c000000)),
// CHECK:     # Relocation 1
// CHECK:     (('word-0', 0x0),
// CHECK:      ('word-1', 0xc000001)),
// CHECK:   ])

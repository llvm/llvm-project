// RUN: llvm-mc -triple x86_64-apple-darwin9 %s -filetype=obj -o - | llvm-readobj -r - | FileCheck %s

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

// CHECK: Relocations [
// CHECK:   Section __text {
// CHECK:     0x0 0 2 1 X86_64_RELOC_SUBTRACTOR 0 _base
// CHECK:     0x0 0 2 1 X86_64_RELOC_UNSIGNED 0 _start_ap_2
// CHECK:   }
// CHECK: ]

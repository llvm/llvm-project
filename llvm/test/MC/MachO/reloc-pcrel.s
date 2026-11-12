// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | llvm-readobj -r - | FileCheck %s

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

// CHECK: Relocations [
// CHECK-NEXT:   Section __text {
// CHECK-NEXT:     0x45 1 2 n/a GENERIC_RELOC_LOCAL_SECTDIFF 1 0x4
// CHECK-NEXT:     0x0 1 2 n/a GENERIC_RELOC_PAIR 1 0x6
// CHECK-NEXT:     0x40 1 2 1 GENERIC_RELOC_VANILLA 0 _c
// CHECK-NEXT:     0x3B 1 2 1 GENERIC_RELOC_VANILLA 0 _c
// CHECK-NEXT:     0x36 1 2 1 GENERIC_RELOC_VANILLA 0 _c
// CHECK-NEXT:     0x31 1 2 n/a GENERIC_RELOC_VANILLA 1 0x4
// CHECK-NEXT:     0x2C 1 2 n/a GENERIC_RELOC_VANILLA 1 0x4
// CHECK-NEXT:     0x27 1 2 0 GENERIC_RELOC_VANILLA 0 __text
// CHECK-NEXT:     0x22 1 2 n/a GENERIC_RELOC_VANILLA 1 0x2
// CHECK-NEXT:     0x1D 1 2 n/a GENERIC_RELOC_VANILLA 1 0x2
// CHECK-NEXT:     0x18 1 2 0 GENERIC_RELOC_VANILLA 0 __text
// CHECK-NEXT:   }
// CHECK-NEXT: ]

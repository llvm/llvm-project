// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | llvm-readobj -r - | FileCheck %s

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

// CHECK: Relocations [
// CHECK-NEXT:   Section __data {
// CHECK-NEXT:     0x10 0 2 n/a GENERIC_RELOC_LOCAL_SECTDIFF 1 0x0
// CHECK-NEXT:     0x0 0 2 n/a GENERIC_RELOC_PAIR 1 0x0
// CHECK-NEXT:     0x8 0 2 n/a GENERIC_RELOC_LOCAL_SECTDIFF 1 0x0
// CHECK-NEXT:     0x0 0 2 n/a GENERIC_RELOC_PAIR 1 0x0
// CHECK-NEXT:   }
// CHECK-NEXT: ]

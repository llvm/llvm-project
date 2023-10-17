// RUN: llvm-mc -triple=aarch64 -filetype=obj %s -o - | \
// RUN:   llvm-readobj -r - | FileCheck %s
.globl func
func:

        .section .data
this:
        .word extern_func@PLT - this + 4
        .word func@plt - . + 8

// CHECK:      Section ({{.*}}) .rela.data
// CHECK-NEXT:   0x0 R_AARCH64_PLT32 extern_func 0x4
// CHECK-NEXT:   0x4 R_AARCH64_PLT32 func 0x8
// CHECK-NEXT: }

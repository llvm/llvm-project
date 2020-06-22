// RUN: llvm-mc -triple=aarch64 -filetype=obj %s -o - | \
// RUN:   llvm-readobj -r | FileCheck %s

        .section .data
this:
        .word extern_func@PLT - this + 4

// CHECK:      Section ({{.*}}) .rela.data
// CHECK-NEXT:   0x0 R_AARCH64_PLT32 extern_func 0x4
// CHECK-NEXT: }

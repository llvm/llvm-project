// REQUIRES: x86-registered-target
// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | not llvm-readobj -r - 2>&1 | FileCheck %s

// CHECK: error: '<stdin>': invalid packed relocation header

.section .rela.dyn, "a", @0x60000001
.ascii "APS9"

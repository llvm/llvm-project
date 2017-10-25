// REQUIRES: x86-registered-target
// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -relocations - | FileCheck %s

// CHECK:      Relocations [
// CHECK-NEXT:   Section (3) .rela.dyn {
// CHECK-NEXT:   }
// CHECK-NEXT: ]

.section .rela.dyn, "a", @0x60000001
.ascii "APS2"
.sleb128 0
.sleb128 0

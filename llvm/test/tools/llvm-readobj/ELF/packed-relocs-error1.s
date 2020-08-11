// REQUIRES: x86-registered-target
// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o %t
// RUN: llvm-readobj --relocations %t 2>&1 | FileCheck %s -DFILE=%t

// CHECK: warning: '[[FILE]]': unable to read relocations from SHT_ANDROID_REL section with index 3: invalid packed relocation header

.section .rela.dyn, "a", @0x60000001
.ascii "APS9"

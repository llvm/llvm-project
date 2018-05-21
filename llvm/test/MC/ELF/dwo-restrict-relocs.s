// RUN: not llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o %t1 -split-dwarf-file %t2 2>&1 | FileCheck %s

// CHECK: error: A relocation may not refer to a dwo section
.quad .foo.dwo

.section .foo.dwo
// CHECK: error: A dwo section may not contain relocations
.quad .text
// CHECK: error: A dwo section may not contain relocations
.quad .foo.dwo

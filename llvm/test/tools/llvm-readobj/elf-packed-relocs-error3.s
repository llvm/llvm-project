// REQUIRES: x86-registered-target
// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | not llvm-readobj -relocations - 2>&1 | FileCheck %s

// CHECK: Error reading file: malformed sleb128, extends past end

.section .rela.dyn, "a", @0x60000001
.ascii "APS2"
.sleb128 4 // Number of relocations
.sleb128 0 // Initial offset

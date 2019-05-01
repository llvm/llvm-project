// REQUIRES: x86-registered-target
// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | not llvm-readobj -r - 2>&1 | FileCheck %s

// CHECK: error: relocation group unexpectedly large

.section .rela.dyn, "a", @0x60000001
.ascii "APS2"
.sleb128 4 // Number of relocations
.sleb128 0 // Initial offset

.sleb128 5 // Number of relocations in group
.sleb128 2 // RELOCATION_GROUPED_BY_OFFSET_DELTA_FLAG
.sleb128 8 // offset delta

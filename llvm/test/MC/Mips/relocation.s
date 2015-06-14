// RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux < %s | llvm-readobj -r | FileCheck  %s

// Test that we produce the correct relocation.
// FIXME: move more relocation only tests here.

        .long foo
// CHECK: R_MIPS_32 foo

        .long foo-.
// CHECK: R_MIPS_PC32 foo

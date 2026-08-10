// Test that an alignment of zero is accepted.
// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o -

  .align 0

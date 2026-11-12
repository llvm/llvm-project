// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o %t
// RUN: llvm-objdump -r %t | FileCheck %s

// Test that we create no fixups for this file since "a" and "b"
// are in the same fragment. If b was in a different section, a
// fixup causing a relocation would be generated in the object file.

// CHECK-NOT: RELOCATION RECORDS

a:
  nop
b:
  .long b - a

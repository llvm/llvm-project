//RUN: not llvm-mc -triple=aarch64-linux -filetype=obj %s -o /dev/null 2> %t2
//RUN: cat %t2 | FileCheck %s

//These tests look for errors that should be reported for invalid object layout
//with the ldr pseudo. They are tested separately from parse errors because they
//only trigger when the file has successfully parsed and the object file is about
//to be written out.

.text
foo:
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: fixup value out of range
  ldr x0, =0x10111
  .space 0xdeadb0

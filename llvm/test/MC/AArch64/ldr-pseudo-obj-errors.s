//RUN: not llvm-mc -triple=aarch64-linux -filetype=obj %s -o %t1 2> %t2
//RUN: cat %t2 | FileCheck %s

//These tests look for errors that should be reported for invalid object layout
//with the ldr pseudo. They are tested separately from parse errors because they
//only trigger when the file has successfully parsed and the object file is about
//to be written out.

.text
foo:
  ldr x0, =0x10111
  .space 0xdeadb0
// CHECK: LVM ERROR: fixup value out of range

@RUN: not llvm-mc -triple=armv7-unknown-linux-gnueabi -filetype=obj %s -o %t1 2> %t2
@RUN: cat %t2 | FileCheck %s
@RUN: not llvm-mc -triple=armv7-darwin-apple -filetype=obj %s -o %t1_darwin 2> %t2_darwin
@RUN: cat %t2_darwin | FileCheck %s

@These tests look for errors that should be reported for invalid object layout
@with the ldr pseudo. They are tested separately from parse errors because they
@only trigger when the file has successfully parsed and the object file is about
@to be written out.

.text
foo:
  ldr r0, =0x12345678
  .space 8000
@ CHECK: error: out of range pc-relative fixup value
@ CHECK: ldr r0, =0x12345678
@ CHECK: ^

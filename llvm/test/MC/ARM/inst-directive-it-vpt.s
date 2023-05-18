// RUN: llvm-mc %s -triple armv8m.main -mattr=+mve -filetype asm -o - | FileCheck %s

  .thumb

// CHECK: it      eq
// CHECK: .inst.n 0x3001
// CHECK: add.w r0, r0, #1
  it eq
  .inst.n 0x3001 // addeq r0, #1
  add r0, #1

// CHECK: vpst
// CHECK: .inst.w 0xef220844
// CHECK: vadd.i32 q0, q1, q2
  vpst
  .inst.w 0xef220844 // vaddt.i32 q0, q1, q2
  vadd.i32 q0, q1, q2

// CHECK: ite eq
// CHECK: .inst.n 0x3001
// CHECK: addne r0, #1
// CHECK: add.w r0, r0, #1
  ite eq
  .inst.n 0x3001 // addeq r0, #1
  addne r0, #1
  add r0, #1

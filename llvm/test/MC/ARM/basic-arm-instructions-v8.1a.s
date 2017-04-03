//RUN: not llvm-mc -triple thumb-none-linux-gnu -mattr=+v8.1a -mattr=neon -show-encoding < %s 2>%t | FileCheck %s --check-prefix=CHECK-V81aTHUMB
//RUN: FileCheck --check-prefix=CHECK-ERROR <%t %s
//RUN: not llvm-mc -triple arm-none-linux-gnu -mattr=+v8.1a -mattr=neon -show-encoding < %s 2>%t | FileCheck %s --check-prefix=CHECK-V81aARM
//RUN: FileCheck --check-prefix=CHECK-ERROR <%t %s

//RUN: not llvm-mc -triple thumb-none-linux-gnu -mattr=+v8 -mattr=neon -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=CHECK-V8
//RUN: not llvm-mc -triple arm-none-linux-gnu -mattr=+v8 -mattr=neon -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=CHECK-V8


  .text
//CHECK-V8THUMB: .text

  vqrdmlah.i8   q0, q1, q2
  vqrdmlah.u16  d0, d1, d2
  vqrdmlsh.f32  q3, q4, q5
  vqrdmlsh.f64  d3, d5, d5

//CHECK-ERROR: error: invalid operand for instruction
//CHECK-ERROR:   vqrdmlah.i8   q0, q1, q2
//CHECK-ERROR:           ^
//CHECK-ERROR: error: invalid operand for instruction
//CHECK-ERROR:   vqrdmlah.u16  d0, d1, d2
//CHECK-ERROR:           ^
//CHECK-ERROR: error: invalid operand for instruction
//CHECK-ERROR:   vqrdmlsh.f32  q3, q4, q5
//CHECK-ERROR:           ^
//CHECK-ERROR: error: invalid operand for instruction
//CHECK-ERROR:   vqrdmlsh.f64  d3, d5, d5
//CHECK-ERROR:           ^
//CHECK-V8: error: invalid operand for instruction
//CHECK-V8:   vqrdmlah.i8   q0, q1, q2
//CHECK-V8:           ^
//CHECK-V8: error: invalid operand for instruction
//CHECK-V8:   vqrdmlah.u16  d0, d1, d2
//CHECK-V8:           ^
//CHECK-V8: error: invalid operand for instruction
//CHECK-V8:   vqrdmlsh.f32  q3, q4, q5
//CHECK-V8:           ^
//CHECK-V8: error: invalid operand for instruction
//CHECK-V8:  vqrdmlsh.f64  d3, d5, d5
//CHECK-V8:           ^

  vqrdmlah.s16    d0, d1, d2
//CHECK-V81aARM:   vqrdmlah.s16  d0, d1, d2      @ encoding: [0x12,0x0b,0x11,0xf3]
//CHECK-V81aTHUMB: vqrdmlah.s16  d0, d1, d2      @ encoding: [0x11,0xff,0x12,0x0b]
//CHECK-V8: error: instruction requires: armv8.1a
//CHECK-V8:  vqrdmlah.s16    d0, d1, d2
//CHECK-V8:  ^

  vqrdmlah.s32  d0, d1, d2
//CHECK-V81aARM:   vqrdmlah.s32  d0, d1, d2      @ encoding: [0x12,0x0b,0x21,0xf3]
//CHECK-V81aTHUMB: vqrdmlah.s32  d0, d1, d2      @ encoding: [0x21,0xff,0x12,0x0b]
//CHECK-V8: error: instruction requires: armv8.1a
//CHECK-V8:  vqrdmlah.s32  d0, d1, d2
//CHECK-V8:  ^

  vqrdmlah.s16  q0, q1, q2
//CHECK-V81aARM:   vqrdmlah.s16  q0, q1, q2      @ encoding: [0x54,0x0b,0x12,0xf3]
//CHECK-V81aTHUMB: vqrdmlah.s16  q0, q1, q2      @ encoding: [0x12,0xff,0x54,0x0b]
//CHECK-V8: error: instruction requires: armv8.1a
//CHECK-V8:  vqrdmlah.s16  q0, q1, q2
//CHECK-V8:  ^

  vqrdmlah.s32  q2, q3, q0
//CHECK-V81aARM:   vqrdmlah.s32  q2, q3, q0      @ encoding: [0x50,0x4b,0x26,0xf3]
//CHECK-V81aTHUMB: vqrdmlah.s32  q2, q3, q0      @ encoding: [0x26,0xff,0x50,0x4b]
//CHECK-V8: error: instruction requires: armv8.1a
//CHECK-V8:  vqrdmlah.s32  q2, q3, q0
//CHECK-V8:  ^


  vqrdmlsh.s16  d7, d6, d5
//CHECK-V81aARM:   vqrdmlsh.s16  d7, d6, d5      @ encoding: [0x15,0x7c,0x16,0xf3]
//CHECK-V81aTHUMB: vqrdmlsh.s16  d7, d6, d5      @ encoding: [0x16,0xff,0x15,0x7c]
//CHECK-V8: error: instruction requires: armv8.1a
//CHECK-V8:  vqrdmlsh.s16  d7, d6, d5
//CHECK-V8:  ^

  vqrdmlsh.s32  d0, d1, d2
//CHECK-V81aARM:   vqrdmlsh.s32  d0, d1, d2      @ encoding: [0x12,0x0c,0x21,0xf3]
//CHECK-V81aTHUMB: vqrdmlsh.s32  d0, d1, d2      @ encoding: [0x21,0xff,0x12,0x0c]
//CHECK-V8: error: instruction requires: armv8.1a
//CHECK-V8:  vqrdmlsh.s32  d0, d1, d2
//CHECK-V8:  ^

  vqrdmlsh.s16  q0, q1, q2
//CHECK-V81aARM:   vqrdmlsh.s16  q0, q1, q2      @ encoding: [0x54,0x0c,0x12,0xf3]
//CHECK-V81aTHUMB: vqrdmlsh.s16  q0, q1, q2      @ encoding: [0x12,0xff,0x54,0x0c]
//CHECK-V8: error: instruction requires: armv8.1a
//CHECK-V8:  vqrdmlsh.s16  q0, q1, q2
//CHECK-V8:  ^

  vqrdmlsh.s32    q3, q4, q5
//CHECK-V81aARM:   vqrdmlsh.s32  q3, q4, q5      @ encoding: [0x5a,0x6c,0x28,0xf3]
//CHECK-V81aTHUMB: vqrdmlsh.s32  q3, q4, q5      @ encoding: [0x28,0xff,0x5a,0x6c]
//CHECK-V8: error: instruction requires: armv8.1a
//CHECK-V8:  vqrdmlsh.s32  q3, q4, q5
//CHECK-V8:  ^


  vqrdmlah.i8   q0, q1, d9[7]
  vqrdmlah.u16  d0, d1, d2[3]
  vqrdmlsh.f32  q3, q4, d5[1]
  vqrdmlsh.f64  d3, d5, d5[0]

//CHECK-ERROR: error: invalid operand for instruction
//CHECK-ERROR:   vqrdmlah.i8   q0, q1, d9[7]
//CHECK-ERROR:           ^
//CHECK-ERROR: error: invalid operand for instruction
//CHECK-ERROR:   vqrdmlah.u16  d0, d1, d2[3]
//CHECK-ERROR:           ^
//CHECK-ERROR: error: invalid operand for instruction
//CHECK-ERROR:   vqrdmlsh.f32  q3, q4, d5[1]
//CHECK-ERROR:           ^
//CHECK-ERROR: error: invalid operand for instruction
//CHECK-ERROR:   vqrdmlsh.f64  d3, d5, d5[0]
//CHECK-ERROR:           ^

  vqrdmlah.s16  d0, d1, d2[0]
//CHECK-V81aARM:   vqrdmlah.s16 d0, d1, d2[0]    @ encoding: [0x42,0x0e,0x91,0xf2]
//CHECK-V81aTHUMB: vqrdmlah.s16  d0, d1, d2[0]   @ encoding: [0x91,0xef,0x42,0x0e]
//CHECK-V8: error: instruction requires: armv8.1a
//CHECK-V8:  vqrdmlah.s16  d0, d1, d2[0]
//CHECK-V8:  ^

  vqrdmlah.s32  d0, d1, d2[0]
//CHECK-V81aARM:   vqrdmlah.s32 d0, d1, d2[0]    @ encoding: [0x42,0x0e,0xa1,0xf2]
//CHECK-V81aTHUMB: vqrdmlah.s32  d0, d1, d2[0]   @ encoding: [0xa1,0xef,0x42,0x0e]
//CHECK-V8: error: instruction requires: armv8.1a
//CHECK-V8:  vqrdmlah.s32  d0, d1, d2[0]
//CHECK-V8:  ^

  vqrdmlah.s16  q0, q1, d2[0]
//CHECK-V81aARM:   vqrdmlah.s16  q0, q1, d2[0]   @ encoding: [0x42,0x0e,0x92,0xf3]
//CHECK-V81aTHUMB: vqrdmlah.s16  q0, q1, d2[0]   @ encoding: [0x92,0xff,0x42,0x0e]
//CHECK-V8: error: instruction requires: armv8.1a
//CHECK-V8:  vqrdmlah.s16  q0, q1, d2[0]
//CHECK-V8:  ^

  vqrdmlah.s32  q0, q1, d2[0]
//CHECK-V81aARM:   vqrdmlah.s32  q0, q1, d2[0]   @ encoding: [0x42,0x0e,0xa2,0xf3]
//CHECK-V81aTHUMB: vqrdmlah.s32  q0, q1, d2[0]   @ encoding: [0xa2,0xff,0x42,0x0e]
//CHECK-V8: error: instruction requires: armv8.1a
//CHECK-V8:  vqrdmlah.s32  q0, q1, d2[0]
//CHECK-V8:  ^


  vqrdmlsh.s16  d0, d1, d2[0]
//CHECK-V81aARM:   vqrdmlsh.s16 d0, d1, d2[0]    @ encoding: [0x42,0x0f,0x91,0xf2]
//CHECK-V81aTHUMB: vqrdmlsh.s16  d0, d1, d2[0]   @ encoding: [0x91,0xef,0x42,0x0f]
//CHECK-V8: error: instruction requires: armv8.1a
//CHECK-V8:  vqrdmlsh.s16  d0, d1, d2[0]
//CHECK-V8:  ^

  vqrdmlsh.s32  d0, d1, d2[0]
//CHECK-V81aARM:   vqrdmlsh.s32 d0, d1, d2[0]    @ encoding: [0x42,0x0f,0xa1,0xf2]
//CHECK-V81aTHUMB: vqrdmlsh.s32  d0, d1, d2[0]   @ encoding: [0xa1,0xef,0x42,0x0f]
//CHECK-V8: error: instruction requires: armv8.1a
//CHECK-V8:  vqrdmlsh.s32  d0, d1, d2[0]
//CHECK-V8:  ^

  vqrdmlsh.s16  q0, q1, d2[0]
//CHECK-V81aARM:   vqrdmlsh.s16 q0, q1, d2[0]    @ encoding: [0x42,0x0f,0x92,0xf3]
//CHECK-V81aTHUMB: vqrdmlsh.s16  q0, q1, d2[0]   @ encoding: [0x92,0xff,0x42,0x0f]
//CHECK-V8: error: instruction requires: armv8.1a
//CHECK-V8:  vqrdmlsh.s16  q0, q1, d2[0]
//CHECK-V8:  ^

  vqrdmlsh.s32  q0, q1, d2[0]
//CHECK-V81aARM:   vqrdmlsh.s32 q0, q1, d2[0]    @ encoding: [0x42,0x0f,0xa2,0xf3]
//CHECK-V81aTHUMB: vqrdmlsh.s32  q0, q1, d2[0]   @ encoding: [0xa2,0xff,0x42,0x0f]
//CHECK-V8: error: instruction requires: armv8.1a
//CHECK-V8:  vqrdmlsh.s32  q0, q1, d2[0]
//CHECK-V8:  ^

  setpan  #0
//CHECK-V81aTHUMB:  setpan  #0                @       encoding: [0x10,0xb6]
//CHECK-V81aARM:    setpan  #0                @       encoding: [0x00,0x00,0x10,0xf1]
//CHECK-V8: error: instruction requires: armv8.1a
//CHECK-V8:  setpan  #0
//CHECK-V8:  ^

  setpan  #1
//CHECK-V81aTHUMB:  setpan  #1                @       encoding: [0x18,0xb6]
//CHECK-V81aARM:    setpan  #1                @       encoding: [0x00,0x02,0x10,0xf1]
//CHECK-V8: error: instruction requires: armv8.1a
//CHECK-V8:  setpan  #1
//CHECK-V8:  ^
  setpan
  setpan #-1
  setpan #2
//CHECK-ERROR: error: too few operands for instruction
//CHECK-ERROR:  setpan
//CHECK-ERROR:  ^
//CHECK-ERROR: error: immediate operand must be in the range [0,1]
//CHECK-ERROR:  setpan #-1
//CHECK-ERROR:         ^
//CHECK-ERROR: error: immediate operand must be in the range [0,1]
//CHECK-ERROR:  setpan #2
//CHECK-ERROR:         ^

  it eq
  setpaneq #0
//CHECK-THUMB-ERROR: error: instruction 'setpan' is not predicable, but condition code specified
//CHECK-THUMB-ERROR:  setpaneq #0
//CHECK-THUMB-ERROR:  ^

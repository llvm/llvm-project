// Test the encoding of shift instructions for the 430X CPU.

// RUN: llvm-mc -triple msp430 -mcpu=msp430x -show-encoding < %s | FileCheck %s
rxxm:
	rrum	#1, r12                         ; encoding: [0x5c,0x03]
	rrum	#2, r12                         ; encoding: [0x5c,0x07]
	rrum	#3, r12                         ; encoding: [0x5c,0x0b]
	rrum	#4, r12                         ; encoding: [0x5c,0x0f]

	rram	#1, r12                         ; encoding: [0x5c,0x01]
	rram	#2, r12                         ; encoding: [0x5c,0x05]
	rram	#3, r12                         ; encoding: [0x5c,0x09]
	rram	#4, r12                         ; encoding: [0x5c,0x0d]

	rlam	#1, r12                         ; encoding: [0x5c,0x02]
	rlam	#2, r12                         ; encoding: [0x5c,0x06]
	rlam	#3, r12                         ; encoding: [0x5c,0x0a]
	rlam	#4, r12                         ; encoding: [0x5c,0x0e]

	rrcm	#1, r12                         ; encoding: [0x5c,0x00]
	rrcm	#2, r12                         ; encoding: [0x5c,0x04]
	rrcm	#3, r12                         ; encoding: [0x5c,0x08]
	rrcm	#4, r12                         ; encoding: [0x5c,0x0c]

// CHECK-LABEL: rxxm:
// CHECK-NEXT: rrum	#1, r12                         ; encoding: [0x5c,0x03]
// CHECK: rrum	#2, r12                         ; encoding: [0x5c,0x07]
// CHECK: rrum	#3, r12                         ; encoding: [0x5c,0x0b]
// CHECK: rrum	#4, r12                         ; encoding: [0x5c,0x0f]

// CHECK: rram	#1, r12                         ; encoding: [0x5c,0x01]
// CHECK: rram	#2, r12                         ; encoding: [0x5c,0x05]
// CHECK: rram	#3, r12                         ; encoding: [0x5c,0x09]
// CHECK: rram	#4, r12                         ; encoding: [0x5c,0x0d]

// CHECK: rlam	#1, r12                         ; encoding: [0x5c,0x02]
// CHECK: rlam	#2, r12                         ; encoding: [0x5c,0x06]
// CHECK: rlam	#3, r12                         ; encoding: [0x5c,0x0a]
// CHECK: rlam	#4, r12                         ; encoding: [0x5c,0x0e]

// CHECK: rrcm	#1, r12                         ; encoding: [0x5c,0x00]
// CHECK: rrcm	#2, r12                         ; encoding: [0x5c,0x04]
// CHECK: rrcm	#3, r12                         ; encoding: [0x5c,0x08]
// CHECK: rrcm	#4, r12                         ; encoding: [0x5c,0x0c]

rrcx:
  rrcx r12

// CHECK-LABEL: rrcx:
// CHECK-NEXT: rrcx	r12                             ; encoding: [0x40,0x18,0x0c,0x10]

rrcx_b:
  rrcx.b r12

// CHECK-LABEL: rrcx_b:
// CHECK-NEXT: rrcx.b	r12                             ; encoding: [0x40,0x18,0x4c,0x10]

rrux:
  rrux r12

// CHECK-LABEL: rrux:
// CHECK-NEXT: rrux	r12                             ; encoding: [0x40,0x19,0x0c,0x10]

rrux_b:
  rrux.b r12

// CHECK-LABEL: rrux_b:
// CHECK-NEXT: rrux.b	r12                             ; encoding: [0x40,0x19,0x4c,0x10]

rrax:
  rrax r12

// CHECK-LABEL: rrax:
// CHECK-NEXT: rrax	r12                             ; encoding: [0x40,0x18,0x0c,0x11]

rrax_b:
  rrax.b r12

// CHECK-LABEL: rrax_b:
// CHECK-NEXT: rrax.b	r12                             ; encoding: [0x40,0x18,0x4c,0x11]

// RLAX is an alias of ADDX with both registers equal. Alias printing is
// disabled, so it prints as ADDX.
rlax:
  rlax r12

// CHECK-LABEL: rlax:
// CHECK-NEXT: addx	r12, r12                        ; encoding: [0x40,0x18,0x0c,0x5c]

rlax_b:
  rlax.b r12

// CHECK-LABEL: rlax_b:
// CHECK-NEXT: addx.b	r12, r12                        ; encoding: [0x40,0x18,0x4c,0x5c]

// ADDX with different registers must not be treated as RLAX.
addx:
  addx r5, r12

// CHECK-LABEL: addx:
// CHECK-NEXT: addx	r5, r12                         ; encoding: [0x40,0x18,0x0c,0x55]

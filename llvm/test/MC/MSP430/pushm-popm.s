// Test the encoding of PUSHM/POPM instructions for the MSP430X CPU.

// RUN: llvm-mc -triple msp430 -mcpu=msp430x -show-encoding < %s | FileCheck %s
pushm:
  pushm #1, r4
  pushm #2, r5
  pushm #3, r6
  pushm #4, r7
  pushm #5, r8
  pushm #6, r9
  pushm #7, r10
  pushm #8, r11
  pushm #9, r12
  pushm #10, r13
  pushm #11, r14
  pushm #12, r15
  pushm #16, r15

  pushm #1, r15
  pushm #2, r15
  pushm #3, r15
  pushm #4, r15
  pushm #5, r15

  pushm #1, r10
  pushm #2, r10
  pushm #3, r10
  pushm #4, r10
  pushm #5, r10

// CHECK-LABEL: pushm:
// CHECK-NEXT: pushm	#1, r4                          ; encoding: [0x04,0x15]
// CHECK: pushm	#2, r5                          ; encoding: [0x15,0x15]
// CHECK: pushm	#3, r6                          ; encoding: [0x26,0x15]
// CHECK: pushm	#4, r7                          ; encoding: [0x37,0x15]
// CHECK: pushm	#5, r8                          ; encoding: [0x48,0x15]
// CHECK: pushm	#6, r9                          ; encoding: [0x59,0x15]
// CHECK: pushm	#7, r10                         ; encoding: [0x6a,0x15]
// CHECK: pushm	#8, r11                         ; encoding: [0x7b,0x15]
// CHECK: pushm	#9, r12                         ; encoding: [0x8c,0x15]
// CHECK: pushm	#10, r13                        ; encoding: [0x9d,0x15]
// CHECK: pushm	#11, r14                        ; encoding: [0xae,0x15]
// CHECK: pushm	#12, r15                        ; encoding: [0xbf,0x15]
// CHECK: pushm	#16, r15                        ; encoding: [0xff,0x15]

// CHECK: pushm	#1, r15                         ; encoding: [0x0f,0x15]
// CHECK: pushm	#2, r15                         ; encoding: [0x1f,0x15]
// CHECK: pushm	#3, r15                         ; encoding: [0x2f,0x15]
// CHECK: pushm	#4, r15                         ; encoding: [0x3f,0x15]
// CHECK: pushm	#5, r15                         ; encoding: [0x4f,0x15]

// CHECK: pushm	#1, r10                         ; encoding: [0x0a,0x15]
// CHECK: pushm	#2, r10                         ; encoding: [0x1a,0x15]
// CHECK: pushm	#3, r10                         ; encoding: [0x2a,0x15]
// CHECK: pushm	#4, r10                         ; encoding: [0x3a,0x15]
// CHECK: pushm	#5, r10                         ; encoding: [0x4a,0x15]

popm:
  popm #1, r4
  popm #2, r5
  popm #3, r6
  popm #4, r7
  popm #5, r8
  popm #6, r9
  popm #7, r10
  popm #8, r11
  popm #9, r12
  popm #10, r13
  popm #11, r14
  popm #12, r15
  popm #16, r15

  popm #1, r15
  popm #2, r15
  popm #3, r15
  popm #4, r15
  popm #5, r15

  popm #1, r10
  popm #2, r10
  popm #3, r10
  popm #4, r10
  popm #5, r10

// CHECK-LABEL: popm:
// CHECK-NEXT: popm	#1, r4                          ; encoding: [0x04,0x17]
// CHECK: popm	#2, r5                          ; encoding: [0x14,0x17]
// CHECK: popm	#3, r6                          ; encoding: [0x24,0x17]
// CHECK: popm	#4, r7                          ; encoding: [0x34,0x17]
// CHECK: popm	#5, r8                          ; encoding: [0x44,0x17]
// CHECK: popm	#6, r9                          ; encoding: [0x54,0x17]
// CHECK: popm	#7, r10                         ; encoding: [0x64,0x17]
// CHECK: popm	#8, r11                         ; encoding: [0x74,0x17]
// CHECK: popm	#9, r12                         ; encoding: [0x84,0x17]
// CHECK: popm	#10, r13                        ; encoding: [0x94,0x17]
// CHECK: popm	#11, r14                        ; encoding: [0xa4,0x17]
// CHECK: popm	#12, r15                        ; encoding: [0xb4,0x17]
// CHECK: popm	#16, r15                        ; encoding: [0xf0,0x17]

// CHECK: popm	#1, r15                         ; encoding: [0x0f,0x17]
// CHECK: popm	#2, r15                         ; encoding: [0x1e,0x17]
// CHECK: popm	#3, r15                         ; encoding: [0x2d,0x17]
// CHECK: popm	#4, r15                         ; encoding: [0x3c,0x17]
// CHECK: popm	#5, r15                         ; encoding: [0x4b,0x17]

// CHECK: popm	#1, r10                         ; encoding: [0x0a,0x17]
// CHECK: popm	#2, r10                         ; encoding: [0x19,0x17]
// CHECK: popm	#3, r10                         ; encoding: [0x28,0x17]
// CHECK: popm	#4, r10                         ; encoding: [0x37,0x17]
// CHECK: popm	#5, r10                         ; encoding: [0x46,0x17]

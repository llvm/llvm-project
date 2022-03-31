// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+streaming-sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// This test file is mostly empty because most 'movprfx' tests are embedded
// with other instructions that are destructive and can be prefixed
// by the movprfx instruction. A list of destructive instructions
// is given below by their mnemonic, which have tests in corresponding
// <mnemonic>.s test files:
//
// abs     decp    fdivr   fnmla   fsubr   mov     sdivr   sqincw  umulh
// add     eon     fmad    fnmls   ftmad   msb     sdot    sqsub   uqadd
// and     eor     fmax    fnmsb   incd    mul     smax    sub     uqdecd
// asr     ext     fmaxnm  frecpx  inch    neg     smin    subr    uqdech
// asrd    fabd    fmin    frinta  incp    not     smulh   sxtb    uqdecp
// asrr    fabs    fminnm  frinti  incw    orn     splice  sxth    uqdecw
// bic     fadd    fmla    frintm  insr    orr     sqadd   sxtw    uqincd
// clasta  fcadd   fmls    frintn  lsl     rbit    sqdecd  uabd    uqinch
// clastb  fcmla   fmov    frintp  lslr    revb    sqdech  ucvtf   uqincp
// cls     fcpy    fmsb    frintx  lsr     revh    sqdecp  udiv    uqincw
// clz     fcvt    fmul    frintz  lsrr    revw    sqdecw  udivr   uqsub
// cnot    fcvtzs  fmulx   fscale  mad     sabd    sqincd  udot    uxtb
// cnt     fcvtzu  fneg    fsqrt   mla     scvtf   sqinch  umax    uxth
// cpy     fdiv    fnmad   fsub    mls     sdiv    sqincp  umin    uxtw


// ------------------------------------------------------------------------- //
// Test compatibility with MOVPRFX instruction with BRK and HLT.
//
// Section 7.1.2 of the SVE Architecture Reference Manual Supplement:
//   "it is permitted to use MOVPRFX to prefix an A64 BRK or HLT instruction"

movprfx z0, z1
// CHECK-INST: movprfx  z0, z1
// CHECK-ENCODING: [0x20,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 bc 20 04 <unknown>

hlt #1
// CHECK-INST: hlt      #0x1
// CHECK-ENCODING: [0x20,0x00,0x40,0xd4]

movprfx z0.d, p0/z, z1.d
// CHECK-INST: movprfx  z0.d, p0/z, z1.d
// CHECK-ENCODING: [0x20,0x20,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 20 d0 04 <unknown>

hlt #1
// CHECK-INST: hlt      #0x1
// CHECK-ENCODING: [0x20,0x00,0x40,0xd4]

movprfx z0, z1
// CHECK-INST: movprfx  z0, z1
// CHECK-ENCODING: [0x20,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 bc 20 04 <unknown>

brk #1
// CHECK-INST: brk      #0x1
// CHECK-ENCODING: [0x20,0x00,0x20,0xd4]

movprfx z0.d, p0/z, z1.d
// CHECK-INST: movprfx  z0.d, p0/z, z1.d
// CHECK-ENCODING: [0x20,0x20,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 20 d0 04 <unknown>

brk #1
// CHECK-INST: brk      #0x1
// CHECK-ENCODING: [0x20,0x00,0x20,0xd4]

// ------------------------------------------------------------------------- //
// Ensure we don't try to apply a prefix to subsequent instructions (upon success)

movprfx z0, z1
// CHECK-INST: movprfx  z0, z1
// CHECK-ENCODING: [0x20,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 bc 20 04 <unknown>

add z0.d, p0/m, z0.d, z1.d
// CHECK-INST: add      z0.d, p0/m, z0.d, z1.d
// CHECK-ENCODING: [0x20,0x00,0xc0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 00 c0 04 <unknown>

add z0.d, p0/m, z0.d, z1.d
// CHECK-INST: add      z0.d, p0/m, z0.d, z1.d
// CHECK-ENCODING: [0x20,0x00,0xc0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 00 c0 04 <unknown>

// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:   | llvm-objdump --no-print-imm-hex -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN

cpy     z0.b, p0/m, w0
// CHECK-INST: mov     z0.b, p0/m, w0
// CHECK-ENCODING: [0x00,0xa0,0x28,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0528a000 <unknown>

cpy     z0.h, p0/m, w0
// CHECK-INST: mov     z0.h, p0/m, w0
// CHECK-ENCODING: [0x00,0xa0,0x68,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0568a000 <unknown>

cpy     z0.s, p0/m, w0
// CHECK-INST: mov     z0.s, p0/m, w0
// CHECK-ENCODING: [0x00,0xa0,0xa8,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05a8a000 <unknown>

cpy     z0.d, p0/m, x0
// CHECK-INST: mov     z0.d, p0/m, x0
// CHECK-ENCODING: [0x00,0xa0,0xe8,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05e8a000 <unknown>

cpy     z31.b, p7/m, wsp
// CHECK-INST: mov     z31.b, p7/m, wsp
// CHECK-ENCODING: [0xff,0xbf,0x28,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0528bfff <unknown>

cpy     z31.h, p7/m, wsp
// CHECK-INST: mov     z31.h, p7/m, wsp
// CHECK-ENCODING: [0xff,0xbf,0x68,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0568bfff <unknown>

cpy     z31.s, p7/m, wsp
// CHECK-INST: mov     z31.s, p7/m, wsp
// CHECK-ENCODING: [0xff,0xbf,0xa8,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05a8bfff <unknown>

cpy     z31.d, p7/m, sp
// CHECK-INST: mov     z31.d, p7/m, sp
// CHECK-ENCODING: [0xff,0xbf,0xe8,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05e8bfff <unknown>

cpy     z0.b, p0/m, b0
// CHECK-INST: mov     z0.b, p0/m, b0
// CHECK-ENCODING: [0x00,0x80,0x20,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05208000 <unknown>

cpy     z31.b, p7/m, b31
// CHECK-INST: mov     z31.b, p7/m, b31
// CHECK-ENCODING: [0xff,0x9f,0x20,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05209fff <unknown>

cpy     z0.h, p0/m, h0
// CHECK-INST: mov     z0.h, p0/m, h0
// CHECK-ENCODING: [0x00,0x80,0x60,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05608000 <unknown>

cpy     z31.h, p7/m, h31
// CHECK-INST: mov     z31.h, p7/m, h31
// CHECK-ENCODING: [0xff,0x9f,0x60,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05609fff <unknown>

cpy     z0.s, p0/m, s0
// CHECK-INST: mov     z0.s, p0/m, s0
// CHECK-ENCODING: [0x00,0x80,0xa0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05a08000 <unknown>

cpy     z31.s, p7/m, s31
// CHECK-INST: mov     z31.s, p7/m, s31
// CHECK-ENCODING: [0xff,0x9f,0xa0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05a09fff <unknown>

cpy     z0.d, p0/m, d0
// CHECK-INST: mov     z0.d, p0/m, d0
// CHECK-ENCODING: [0x00,0x80,0xe0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05e08000 <unknown>

cpy     z31.d, p7/m, d31
// CHECK-INST: mov     z31.d, p7/m, d31
// CHECK-ENCODING: [0xff,0x9f,0xe0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05e09fff <unknown>

cpy     z5.b, p0/z, #-128
// CHECK-INST: mov     z5.b, p0/z, #-128
// CHECK-ENCODING: [0x05,0x10,0x10,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05101005  <unknown>

cpy     z5.b, p0/z, #127
// CHECK-INST: mov     z5.b, p0/z, #127
// CHECK-ENCODING: [0xe5,0x0f,0x10,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05100fe5  <unknown>

cpy     z5.b, p0/z, #255
// CHECK-INST: mov     z5.b, p0/z, #-1
// CHECK-ENCODING: [0xe5,0x1f,0x10,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05101fe5  <unknown>

cpy     z21.h, p0/z, #-128
// CHECK-INST: mov     z21.h, p0/z, #-128
// CHECK-ENCODING: [0x15,0x10,0x50,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05501015  <unknown>

cpy     z21.h, p0/z, #-128, lsl #8
// CHECK-INST: mov     z21.h, p0/z, #-32768
// CHECK-ENCODING: [0x15,0x30,0x50,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05503015  <unknown>

cpy     z21.h, p0/z, #-32768
// CHECK-INST: mov     z21.h, p0/z, #-32768
// CHECK-ENCODING: [0x15,0x30,0x50,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05503015  <unknown>

cpy     z21.h, p0/z, #127
// CHECK-INST: mov     z21.h, p0/z, #127
// CHECK-ENCODING: [0xf5,0x0f,0x50,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05500ff5  <unknown>

cpy     z21.h, p0/z, #127, lsl #8
// CHECK-INST: mov     z21.h, p0/z, #32512
// CHECK-ENCODING: [0xf5,0x2f,0x50,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05502ff5  <unknown>

cpy     z21.h, p0/z, #32512
// CHECK-INST: mov     z21.h, p0/z, #32512
// CHECK-ENCODING: [0xf5,0x2f,0x50,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05502ff5  <unknown>

cpy     z21.s, p0/z, #-128
// CHECK-INST: mov     z21.s, p0/z, #-128
// CHECK-ENCODING: [0x15,0x10,0x90,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05901015  <unknown>

cpy     z21.s, p0/z, #-128, lsl #8
// CHECK-INST: mov     z21.s, p0/z, #-32768
// CHECK-ENCODING: [0x15,0x30,0x90,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05903015  <unknown>

cpy     z21.s, p0/z, #-32768
// CHECK-INST: mov     z21.s, p0/z, #-32768
// CHECK-ENCODING: [0x15,0x30,0x90,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05903015  <unknown>

cpy     z21.s, p0/z, #127
// CHECK-INST: mov     z21.s, p0/z, #127
// CHECK-ENCODING: [0xf5,0x0f,0x90,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05900ff5  <unknown>

cpy     z21.s, p0/z, #127, lsl #8
// CHECK-INST: mov     z21.s, p0/z, #32512
// CHECK-ENCODING: [0xf5,0x2f,0x90,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05902ff5  <unknown>

cpy     z21.s, p0/z, #32512
// CHECK-INST: mov     z21.s, p0/z, #32512
// CHECK-ENCODING: [0xf5,0x2f,0x90,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05902ff5  <unknown>

cpy     z21.d, p0/z, #-128
// CHECK-INST: mov     z21.d, p0/z, #-128
// CHECK-ENCODING: [0x15,0x10,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d01015  <unknown>

cpy     z21.d, p0/z, #-128, lsl #8
// CHECK-INST: mov     z21.d, p0/z, #-32768
// CHECK-ENCODING: [0x15,0x30,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d03015  <unknown>

cpy     z21.d, p0/z, #-32768
// CHECK-INST: mov     z21.d, p0/z, #-32768
// CHECK-ENCODING: [0x15,0x30,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d03015  <unknown>

cpy     z21.d, p0/z, #127
// CHECK-INST: mov     z21.d, p0/z, #127
// CHECK-ENCODING: [0xf5,0x0f,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d00ff5  <unknown>

cpy     z21.d, p0/z, #127, lsl #8
// CHECK-INST: mov     z21.d, p0/z, #32512
// CHECK-ENCODING: [0xf5,0x2f,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d02ff5  <unknown>

cpy     z21.d, p0/z, #32512
// CHECK-INST: mov     z21.d, p0/z, #32512
// CHECK-ENCODING: [0xf5,0x2f,0xd0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d02ff5  <unknown>

// --------------------------------------------------------------------------//
// Tests where the negative immediate is in bounds when interpreted
// as the element type.

cpy z0.b, p0/z, #-129
// CHECK-INST: mov     z0.b, p0/z, #127
// CHECK-ENCODING: [0xe0,0x0f,0x10,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05100fe0  <unknown>

cpy z0.h, p0/z, #-33024
// CHECK-INST: mov     z0.h, p0/z, #32512
// CHECK-ENCODING: [0xe0,0x2f,0x50,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05502fe0  <unknown>

cpy z0.h, p0/z, #-129, lsl #8
// CHECK-INST: mov     z0.h, p0/z, #32512
// CHECK-ENCODING: [0xe0,0x2f,0x50,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05502fe0  <unknown>


// --------------------------------------------------------------------------//
// Tests for merging variant (/m) and testing the range of predicate (> 7)
// is allowed.

cpy     z5.b, p15/m, #-128
// CHECK-INST: mov     z5.b, p15/m, #-128
// CHECK-ENCODING: [0x05,0x50,0x1f,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 051f5005  <unknown>

cpy     z21.h, p15/m, #-128
// CHECK-INST: mov     z21.h, p15/m, #-128
// CHECK-ENCODING: [0x15,0x50,0x5f,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 055f5015  <unknown>

cpy     z21.h, p15/m, #-128, lsl #8
// CHECK-INST: mov     z21.h, p15/m, #-32768
// CHECK-ENCODING: [0x15,0x70,0x5f,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 055f7015  <unknown>

cpy     z21.s, p15/m, #-128
// CHECK-INST: mov     z21.s, p15/m, #-128
// CHECK-ENCODING: [0x15,0x50,0x9f,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 059f5015  <unknown>

cpy     z21.s, p15/m, #-128, lsl #8
// CHECK-INST: mov     z21.s, p15/m, #-32768
// CHECK-ENCODING: [0x15,0x70,0x9f,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 059f7015  <unknown>

cpy     z21.d, p15/m, #-128
// CHECK-INST: mov     z21.d, p15/m, #-128
// CHECK-ENCODING: [0x15,0x50,0xdf,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05df5015  <unknown>

cpy     z21.d, p15/m, #-128, lsl #8
// CHECK-INST: mov     z21.d, p15/m, #-32768
// CHECK-ENCODING: [0x15,0x70,0xdf,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05df7015  <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z31.d, p7/z, z6.d
// CHECK-INST: movprfx	z31.d, p7/z, z6.d
// CHECK-ENCODING: [0xdf,0x3c,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04d03cdf <unknown>

cpy     z31.d, p7/m, sp
// CHECK-INST: mov	z31.d, p7/m, sp
// CHECK-ENCODING: [0xff,0xbf,0xe8,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05e8bfff <unknown>

movprfx z31, z6
// CHECK-INST: movprfx	z31, z6
// CHECK-ENCODING: [0xdf,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bcdf <unknown>

cpy     z31.d, p7/m, sp
// CHECK-INST: mov	z31.d, p7/m, sp
// CHECK-ENCODING: [0xff,0xbf,0xe8,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05e8bfff <unknown>

movprfx z21.d, p7/z, z28.d
// CHECK-INST: movprfx	z21.d, p7/z, z28.d
// CHECK-ENCODING: [0x95,0x3f,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04d03f95 <unknown>

cpy     z21.d, p7/m, #-128, lsl #8
// CHECK-INST: mov	z21.d, p7/m, #-32768
// CHECK-ENCODING: [0x15,0x70,0xd7,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05d77015 <unknown>

movprfx z21, z28
// CHECK-INST: movprfx	z21, z28
// CHECK-ENCODING: [0x95,0xbf,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bf95 <unknown>

cpy     z21.d, p15/m, #-128, lsl #8
// CHECK-INST: mov	z21.d, p15/m, #-32768
// CHECK-ENCODING: [0x15,0x70,0xdf,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05df7015 <unknown>

movprfx z4.d, p7/z, z6.d
// CHECK-INST: movprfx	z4.d, p7/z, z6.d
// CHECK-ENCODING: [0xc4,0x3c,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04d03cc4 <unknown>

cpy     z4.d, p7/m, d31
// CHECK-INST: mov	z4.d, p7/m, d31
// CHECK-ENCODING: [0xe4,0x9f,0xe0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05e09fe4 <unknown>

movprfx z4, z6
// CHECK-INST: movprfx	z4, z6
// CHECK-ENCODING: [0xc4,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bcc4 <unknown>

cpy     z4.d, p7/m, d31
// CHECK-INST: mov	z4.d, p7/m, d31
// CHECK-ENCODING: [0xe4,0x9f,0xe0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05e09fe4 <unknown>

// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p3 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p3 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2p3 --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p3 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2p3 --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p3 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2p3 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// ----------------------------------------------------------
// Lookup table read with 6-bit indices (single)

luti6 z0.b, zt0, z0
// CHECK-INST: luti6 z0.b, zt0, z0
// CHECK-ENCODING: encoding: [0x00,0x40,0xc8,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c0c84000 <unknown>

luti6 z31.b, zt0, z0
// CHECK-INST: luti6 z31.b, zt0, z0
// CHECK-ENCODING: encoding: [0x1f,0x40,0xc8,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c0c8401f <unknown>

luti6 z0.b, zt0, z31
// CHECK-INST: luti6 z0.b, zt0, z31
// CHECK-ENCODING: encoding: [0xe0,0x43,0xc8,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c0c843e0 <unknown>

luti6 z31.b, zt0, z31
// CHECK-INST: luti6 z31.b, zt0, z31
// CHECK-ENCODING: encoding: [0xff,0x43,0xc8,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c0c843ff <unknown>

// ----------------------------------------------------------
// Lookup table read with 6-bit indices (16-bit) - consecutive

luti6 { z0.h - z3.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK-INST: luti6 { z0.h - z3.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK-ENCODING: encoding: [0x00,0xf4,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c120f400 <unknown>

luti6 { z8.h - z11.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK-INST: luti6 { z8.h - z11.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK-ENCODING: encoding: [0x08,0xf4,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c120f408 <unknown>

luti6 { z20.h - z23.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK-INST: luti6 { z20.h - z23.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK-ENCODING: encoding: [0x14,0xf4,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c120f414 <unknown>

luti6 { z28.h - z31.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK-INST: luti6 { z28.h - z31.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK-ENCODING: encoding: [0x1c,0xf4,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c120f41c <unknown>

luti6 { z0.h - z3.h }, { z31.h, z0.h }, { z31, z0 }[1]
// CHECK-INST: luti6 { z0.h - z3.h }, { z31.h, z0.h }, { z31, z0 }[1]
// CHECK-ENCODING: encoding: [0xe0,0xf7,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c17ff7e0 <unknown>

luti6 { z8.h - z11.h }, { z31.h, z0.h }, { z31, z0 }[1]
// CHECK-INST: luti6 { z8.h - z11.h }, { z31.h, z0.h }, { z31, z0 }[1]
// CHECK-ENCODING: encoding: [0xe8,0xf7,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c17ff7e8 <unknown>

luti6 { z20.h - z23.h }, { z31.h, z0.h }, { z31, z0 }[1]
// CHECK-INST: luti6 { z20.h - z23.h }, { z31.h, z0.h }, { z31, z0 }[1]
// CHECK-ENCODING: encoding: [0xf4,0xf7,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c17ff7f4 <unknown>

luti6 { z28.h - z31.h }, { z31.h, z0.h }, { z31, z0 }[1]
// CHECK-INST: luti6 { z28.h - z31.h }, { z31.h, z0.h }, { z31, z0 }[1]
// CHECK-ENCODING: encoding: [0xfc,0xf7,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c17ff7fc <unknown>

// ----------------------------------------------------------
// Lookup table read with 6-bit indices (16-bit) - strided

luti6 { z0.h, z4.h, z8.h, z12.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK-INST: luti6 { z0.h, z4.h, z8.h, z12.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK-ENCODING: encoding: [0x00,0xfc,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c120fc00 <unknown>

luti6 { z1.h, z5.h, z9.h, z13.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK-INST: luti6 { z1.h, z5.h, z9.h, z13.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK-ENCODING: encoding: [0x01,0xfc,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c120fc01 <unknown>

luti6 { z2.h, z6.h, z10.h, z14.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK-INST: luti6 { z2.h, z6.h, z10.h, z14.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK-ENCODING: encoding: [0x02,0xfc,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c120fc02 <unknown>

luti6 { z3.h, z7.h, z11.h, z15.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK-INST: luti6 { z3.h, z7.h, z11.h, z15.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK-ENCODING: encoding: [0x03,0xfc,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c120fc03 <unknown>

luti6 { z16.h, z20.h, z24.h, z28.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK-INST: luti6 { z16.h, z20.h, z24.h, z28.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK-ENCODING: encoding: [0x10,0xfc,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c120fc10 <unknown>

luti6 { z17.h, z21.h, z25.h, z29.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK-INST: luti6 { z17.h, z21.h, z25.h, z29.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK-ENCODING: encoding: [0x11,0xfc,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c120fc11 <unknown>

luti6 { z18.h, z22.h, z26.h, z30.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK-INST: luti6 { z18.h, z22.h, z26.h, z30.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK-ENCODING: encoding: [0x12,0xfc,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c120fc12 <unknown>

luti6 { z19.h, z23.h, z27.h, z31.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK-INST: luti6 { z19.h, z23.h, z27.h, z31.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK-ENCODING: encoding: [0x13,0xfc,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c120fc13 <unknown>

luti6 { z0.h, z4.h, z8.h, z12.h }, { z31.h, z0.h }, { z31, z0 }[1]
// CHECK-INST: luti6 { z0.h, z4.h, z8.h, z12.h }, { z31.h, z0.h }, { z31, z0 }[1]
// CHECK-ENCODING: encoding: [0xe0,0xff,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c17fffe0 <unknown>

luti6 { z1.h, z5.h, z9.h, z13.h }, { z31.h, z0.h }, { z31, z0 }[1]
// CHECK-INST: luti6 { z1.h, z5.h, z9.h, z13.h }, { z31.h, z0.h }, { z31, z0 }[1]
// CHECK-ENCODING: encoding: [0xe1,0xff,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c17fffe1 <unknown>

luti6 { z2.h, z6.h, z10.h, z14.h }, { z31.h, z0.h }, { z31, z0 }[1]
// CHECK-INST: luti6 { z2.h, z6.h, z10.h, z14.h }, { z31.h, z0.h }, { z31, z0 }[1]
// CHECK-ENCODING: encoding: [0xe2,0xff,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c17fffe2 <unknown>

luti6 { z3.h, z7.h, z11.h, z15.h }, { z31.h, z0.h }, { z31, z0 }[1]
// CHECK-INST: luti6 { z3.h, z7.h, z11.h, z15.h }, { z31.h, z0.h }, { z31, z0 }[1]
// CHECK-ENCODING: encoding: [0xe3,0xff,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c17fffe3 <unknown>

luti6 { z16.h, z20.h, z24.h, z28.h }, { z31.h, z0.h }, { z31, z0 }[1]
// CHECK-INST: luti6 { z16.h, z20.h, z24.h, z28.h }, { z31.h, z0.h }, { z31, z0 }[1]
// CHECK-ENCODING: encoding: [0xf0,0xff,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c17ffff0 <unknown>

luti6 { z17.h, z21.h, z25.h, z29.h }, { z31.h, z0.h }, { z31, z0 }[1]
// CHECK-INST: luti6 { z17.h, z21.h, z25.h, z29.h }, { z31.h, z0.h }, { z31, z0 }[1]
// CHECK-ENCODING: encoding: [0xf1,0xff,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c17ffff1 <unknown>

luti6 { z18.h, z22.h, z26.h, z30.h }, { z31.h, z0.h }, { z31, z0 }[1]
// CHECK-INST: luti6 { z18.h, z22.h, z26.h, z30.h }, { z31.h, z0.h }, { z31, z0 }[1]
// CHECK-ENCODING: encoding: [0xf2,0xff,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c17ffff2 <unknown>

luti6 { z19.h, z23.h, z27.h, z31.h }, { z31.h, z0.h }, { z31, z0 }[1]
// CHECK-INST: luti6 { z19.h, z23.h, z27.h, z31.h }, { z31.h, z0.h }, { z31, z0 }[1]
// CHECK-ENCODING: encoding: [0xf3,0xff,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c17ffff3 <unknown>

// ----------------------------------------------------------
// Lookup table read with 6-bit indices (8-bit) - consecutive

luti6 { z8.b - z11.b }, zt0, { z0 - z2 }
// CHECK-INST: luti6 { z8.b - z11.b }, zt0, { z0 - z2 }
// CHECK-ENCODING: encoding: [0x08,0x00,0x8a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c08a0008 <unknown>

luti6 { z20.b - z23.b }, zt0, { z0 - z2 }
// CHECK-INST: luti6 { z20.b - z23.b }, zt0, { z0 - z2 }
// CHECK-ENCODING: encoding: [0x14,0x00,0x8a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c08a0014 <unknown>

luti6 { z28.b - z31.b }, zt0, { z0 - z2 }
// CHECK-INST: luti6 { z28.b - z31.b }, zt0, { z0 - z2 }
// CHECK-ENCODING: encoding: [0x1c,0x00,0x8a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c08a001c <unknown>

luti6 { z0.b - z3.b }, zt0, { z2 - z4 }
// CHECK-INST: luti6 { z0.b - z3.b }, zt0, { z2 - z4 }
// CHECK-ENCODING: encoding: [0x00,0x01,0x8a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c08a0100 <unknown>

luti6 { z8.b - z11.b }, zt0, { z2 - z4 }
// CHECK-INST: luti6 { z8.b - z11.b }, zt0, { z2 - z4 }
// CHECK-ENCODING: encoding: [0x08,0x01,0x8a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c08a0108 <unknown>

luti6 { z20.b - z23.b }, zt0, { z2 - z4 }
// CHECK-INST: luti6 { z20.b - z23.b }, zt0, { z2 - z4 }
// CHECK-ENCODING: encoding: [0x14,0x01,0x8a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c08a0114 <unknown>

luti6 { z28.b - z31.b }, zt0, { z2 - z4 }
// CHECK-INST: luti6 { z28.b - z31.b }, zt0, { z2 - z4 }
// CHECK-ENCODING: encoding: [0x1c,0x01,0x8a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c08a011c <unknown>

luti6 { z0.b - z3.b }, zt0, { z5 - z7 }
// CHECK-INST: luti6 { z0.b - z3.b }, zt0, { z5 - z7 }
// CHECK-ENCODING: encoding: [0x80,0x02,0x8a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c08a0280 <unknown>

luti6 { z8.b - z11.b }, zt0, { z5 - z7 }
// CHECK-INST: luti6 { z8.b - z11.b }, zt0, { z5 - z7 }
// CHECK-ENCODING: encoding: [0x88,0x02,0x8a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c08a0288 <unknown>

luti6 { z20.b - z23.b }, zt0, { z5 - z7 }
// CHECK-INST: luti6 { z20.b - z23.b }, zt0, { z5 - z7 }
// CHECK-ENCODING: encoding: [0x94,0x02,0x8a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c08a0294 <unknown>

luti6 { z28.b - z31.b }, zt0, { z5 - z7 }
// CHECK-INST: luti6 { z28.b - z31.b }, zt0, { z5 - z7 }
// CHECK-ENCODING: encoding: [0x9c,0x02,0x8a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c08a029c <unknown>

luti6 { z0.b - z3.b }, zt0, { z7 - z9 }
// CHECK-INST: luti6 { z0.b - z3.b }, zt0, { z7 - z9 }
// CHECK-ENCODING: encoding: [0x80,0x03,0x8a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c08a0380 <unknown>

luti6 { z8.b - z11.b }, zt0, { z7 - z9 }
// CHECK-INST: luti6 { z8.b - z11.b }, zt0, { z7 - z9 }
// CHECK-ENCODING: encoding: [0x88,0x03,0x8a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c08a0388 <unknown>

luti6 { z20.b - z23.b }, zt0, { z7 - z9 }
// CHECK-INST: luti6 { z20.b - z23.b }, zt0, { z7 - z9 }
// CHECK-ENCODING: encoding: [0x94,0x03,0x8a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c08a0394 <unknown>

luti6 { z28.b - z31.b }, zt0, { z7 - z9 }
// CHECK-INST: luti6 { z28.b - z31.b }, zt0, { z7 - z9 }
// CHECK-ENCODING: encoding: [0x9c,0x03,0x8a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c08a039c <unknown>

// ----------------------------------------------------------
// Lookup table read with 6-bit indices (8-bit) - strided

luti6 { z1.b, z5.b, z9.b, z13.b }, zt0, { z0 - z2 }
// CHECK-INST: luti6 { z1.b, z5.b, z9.b, z13.b }, zt0, { z0 - z2 }
// CHECK-ENCODING: encoding: [0x01,0x00,0x9a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c09a0001 <unknown>

luti6 { z2.b, z6.b, z10.b, z14.b }, zt0, { z0 - z2 }
// CHECK-INST: luti6 { z2.b, z6.b, z10.b, z14.b }, zt0, { z0 - z2 }
// CHECK-ENCODING: encoding: [0x02,0x00,0x9a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c09a0002 <unknown>

luti6 { z3.b, z7.b, z11.b, z15.b }, zt0, { z0 - z2 }
// CHECK-INST: luti6 { z3.b, z7.b, z11.b, z15.b }, zt0, { z0 - z2 }
// CHECK-ENCODING: encoding: [0x03,0x00,0x9a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c09a0003 <unknown>

luti6 { z16.b, z20.b, z24.b, z28.b }, zt0, { z0 - z2 }
// CHECK-INST: luti6 { z16.b, z20.b, z24.b, z28.b }, zt0, { z0 - z2 }
// CHECK-ENCODING: encoding: [0x10,0x00,0x9a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c09a0010 <unknown>

luti6 { z17.b, z21.b, z25.b, z29.b }, zt0, { z0 - z2 }
// CHECK-INST: luti6 { z17.b, z21.b, z25.b, z29.b }, zt0, { z0 - z2 }
// CHECK-ENCODING: encoding: [0x11,0x00,0x9a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c09a0011 <unknown>

luti6 { z18.b, z22.b, z26.b, z30.b }, zt0, { z0 - z2 }
// CHECK-INST: luti6 { z18.b, z22.b, z26.b, z30.b }, zt0, { z0 - z2 }
// CHECK-ENCODING: encoding: [0x12,0x00,0x9a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c09a0012 <unknown>

luti6 { z19.b, z23.b, z27.b, z31.b }, zt0, { z0 - z2 }
// CHECK-INST: luti6 { z19.b, z23.b, z27.b, z31.b }, zt0, { z0 - z2 }
// CHECK-ENCODING: encoding: [0x13,0x00,0x9a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c09a0013 <unknown>

luti6 { z0.b, z4.b, z8.b, z12.b }, zt0, { z2 - z4 }
// CHECK-INST: luti6 { z0.b, z4.b, z8.b, z12.b }, zt0, { z2 - z4 }
// CHECK-ENCODING: encoding: [0x00,0x01,0x9a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c09a0100 <unknown>

luti6 { z1.b, z5.b, z9.b, z13.b }, zt0, { z2 - z4 }
// CHECK-INST: luti6 { z1.b, z5.b, z9.b, z13.b }, zt0, { z2 - z4 }
// CHECK-ENCODING: encoding: [0x01,0x01,0x9a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c09a0101 <unknown>

luti6 { z2.b, z6.b, z10.b, z14.b }, zt0, { z2 - z4 }
// CHECK-INST: luti6 { z2.b, z6.b, z10.b, z14.b }, zt0, { z2 - z4 }
// CHECK-ENCODING: encoding: [0x02,0x01,0x9a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c09a0102 <unknown>

luti6 { z3.b, z7.b, z11.b, z15.b }, zt0, { z2 - z4 }
// CHECK-INST: luti6 { z3.b, z7.b, z11.b, z15.b }, zt0, { z2 - z4 }
// CHECK-ENCODING: encoding: [0x03,0x01,0x9a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c09a0103 <unknown>

luti6 { z16.b, z20.b, z24.b, z28.b }, zt0, { z2 - z4 }
// CHECK-INST: luti6 { z16.b, z20.b, z24.b, z28.b }, zt0, { z2 - z4 }
// CHECK-ENCODING: encoding: [0x10,0x01,0x9a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c09a0110 <unknown>

luti6 { z17.b, z21.b, z25.b, z29.b }, zt0, { z2 - z4 }
// CHECK-INST: luti6 { z17.b, z21.b, z25.b, z29.b }, zt0, { z2 - z4 }
// CHECK-ENCODING: encoding: [0x11,0x01,0x9a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c09a0111 <unknown>

luti6 { z18.b, z22.b, z26.b, z30.b }, zt0, { z2 - z4 }
// CHECK-INST: luti6 { z18.b, z22.b, z26.b, z30.b }, zt0, { z2 - z4 }
// CHECK-ENCODING: encoding: [0x12,0x01,0x9a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c09a0112 <unknown>

luti6 { z19.b, z23.b, z27.b, z31.b }, zt0, { z2 - z4 }
// CHECK-INST: luti6 { z19.b, z23.b, z27.b, z31.b }, zt0, { z2 - z4 }
// CHECK-ENCODING: encoding: [0x13,0x01,0x9a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c09a0113 <unknown>

luti6 { z0.b, z4.b, z8.b, z12.b }, zt0, { z5 - z7 }
// CHECK-INST: luti6 { z0.b, z4.b, z8.b, z12.b }, zt0, { z5 - z7 }
// CHECK-ENCODING: encoding: [0x80,0x02,0x9a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c09a0280 <unknown>

luti6 { z1.b, z5.b, z9.b, z13.b }, zt0, { z5 - z7 }
// CHECK-INST: luti6 { z1.b, z5.b, z9.b, z13.b }, zt0, { z5 - z7 }
// CHECK-ENCODING: encoding: [0x81,0x02,0x9a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c09a0281 <unknown>

luti6 { z2.b, z6.b, z10.b, z14.b }, zt0, { z5 - z7 }
// CHECK-INST: luti6 { z2.b, z6.b, z10.b, z14.b }, zt0, { z5 - z7 }
// CHECK-ENCODING: encoding: [0x82,0x02,0x9a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c09a0282 <unknown>

luti6 { z3.b, z7.b, z11.b, z15.b }, zt0, { z5 - z7 }
// CHECK-INST: luti6 { z3.b, z7.b, z11.b, z15.b }, zt0, { z5 - z7 }
// CHECK-ENCODING: encoding: [0x83,0x02,0x9a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c09a0283 <unknown>

luti6 { z16.b, z20.b, z24.b, z28.b }, zt0, { z5 - z7 }
// CHECK-INST: luti6 { z16.b, z20.b, z24.b, z28.b }, zt0, { z5 - z7 }
// CHECK-ENCODING: encoding: [0x90,0x02,0x9a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c09a0290 <unknown>

luti6 { z17.b, z21.b, z25.b, z29.b }, zt0, { z5 - z7 }
// CHECK-INST: luti6 { z17.b, z21.b, z25.b, z29.b }, zt0, { z5 - z7 }
// CHECK-ENCODING: encoding: [0x91,0x02,0x9a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c09a0291 <unknown>

luti6 { z18.b, z22.b, z26.b, z30.b }, zt0, { z5 - z7 }
// CHECK-INST: luti6 { z18.b, z22.b, z26.b, z30.b }, zt0, { z5 - z7 }
// CHECK-ENCODING: encoding: [0x92,0x02,0x9a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c09a0292 <unknown>

luti6 { z19.b, z23.b, z27.b, z31.b }, zt0, { z5 - z7 }
// CHECK-INST: luti6 { z19.b, z23.b, z27.b, z31.b }, zt0, { z5 - z7 }
// CHECK-ENCODING: encoding: [0x93,0x02,0x9a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c09a0293 <unknown>

luti6 { z0.b, z4.b, z8.b, z12.b }, zt0, { z7 - z9 }
// CHECK-INST: luti6 { z0.b, z4.b, z8.b, z12.b }, zt0, { z7 - z9 }
// CHECK-ENCODING: encoding: [0x80,0x03,0x9a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c09a0380 <unknown>

luti6 { z1.b, z5.b, z9.b, z13.b }, zt0, { z7 - z9 }
// CHECK-INST: luti6 { z1.b, z5.b, z9.b, z13.b }, zt0, { z7 - z9 }
// CHECK-ENCODING: encoding: [0x81,0x03,0x9a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c09a0381 <unknown>

luti6 { z2.b, z6.b, z10.b, z14.b }, zt0, { z7 - z9 }
// CHECK-INST: luti6 { z2.b, z6.b, z10.b, z14.b }, zt0, { z7 - z9 }
// CHECK-ENCODING: encoding: [0x82,0x03,0x9a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c09a0382 <unknown>

luti6 { z3.b, z7.b, z11.b, z15.b }, zt0, { z7 - z9 }
// CHECK-INST: luti6 { z3.b, z7.b, z11.b, z15.b }, zt0, { z7 - z9 }
// CHECK-ENCODING: encoding: [0x83,0x03,0x9a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c09a0383 <unknown>

luti6 { z16.b, z20.b, z24.b, z28.b }, zt0, { z7 - z9 }
// CHECK-INST: luti6 { z16.b, z20.b, z24.b, z28.b }, zt0, { z7 - z9 }
// CHECK-ENCODING: encoding: [0x90,0x03,0x9a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c09a0390 <unknown>

luti6 { z17.b, z21.b, z25.b, z29.b }, zt0, { z7 - z9 }
// CHECK-INST: luti6 { z17.b, z21.b, z25.b, z29.b }, zt0, { z7 - z9 }
// CHECK-ENCODING: encoding: [0x91,0x03,0x9a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c09a0391 <unknown>

luti6 { z18.b, z22.b, z26.b, z30.b }, zt0, { z7 - z9 }
// CHECK-INST: luti6 { z18.b, z22.b, z26.b, z30.b }, zt0, { z7 - z9 }
// CHECK-ENCODING: encoding: [0x92,0x03,0x9a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c09a0392 <unknown>

luti6 { z19.b, z23.b, z27.b, z31.b }, zt0, { z7 - z9 }
// CHECK-INST: luti6 { z19.b, z23.b, z27.b, z31.b }, zt0, { z7 - z9 }
// CHECK-ENCODING: encoding: [0x93,0x03,0x9a,0xc0]
// CHECK-ERROR: instruction requires: sme2p3
// CHECK-UNKNOWN: c09a0393 <unknown>

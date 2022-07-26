// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme < %s \
// RUN:        | llvm-objdump -d --mattr=+sme - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme < %s \
// RUN:   | llvm-objdump -d --mattr=-sme - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

sumops  za0.s, p0/m, p0/m, z0.b, z0.b
// CHECK-INST: sumops  za0.s, p0/m, p0/m, z0.b, z0.b
// CHECK-ENCODING: [0x10,0x00,0xa0,0xa0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a0a00010 <unknown>

sumops  za1.s, p5/m, p2/m, z10.b, z21.b
// CHECK-INST: sumops  za1.s, p5/m, p2/m, z10.b, z21.b
// CHECK-ENCODING: [0x51,0x55,0xb5,0xa0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a0b55551 <unknown>

sumops  za3.s, p3/m, p7/m, z13.b, z8.b
// CHECK-INST: sumops  za3.s, p3/m, p7/m, z13.b, z8.b
// CHECK-ENCODING: [0xb3,0xed,0xa8,0xa0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a0a8edb3 <unknown>

sumops  za3.s, p7/m, p7/m, z31.b, z31.b
// CHECK-INST: sumops  za3.s, p7/m, p7/m, z31.b, z31.b
// CHECK-ENCODING: [0xf3,0xff,0xbf,0xa0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a0bffff3 <unknown>

sumops  za1.s, p3/m, p0/m, z17.b, z16.b
// CHECK-INST: sumops  za1.s, p3/m, p0/m, z17.b, z16.b
// CHECK-ENCODING: [0x31,0x0e,0xb0,0xa0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a0b00e31 <unknown>

sumops  za1.s, p1/m, p4/m, z1.b, z30.b
// CHECK-INST: sumops  za1.s, p1/m, p4/m, z1.b, z30.b
// CHECK-ENCODING: [0x31,0x84,0xbe,0xa0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a0be8431 <unknown>

sumops  za0.s, p5/m, p2/m, z19.b, z20.b
// CHECK-INST: sumops  za0.s, p5/m, p2/m, z19.b, z20.b
// CHECK-ENCODING: [0x70,0x56,0xb4,0xa0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a0b45670 <unknown>

sumops  za0.s, p6/m, p0/m, z12.b, z2.b
// CHECK-INST: sumops  za0.s, p6/m, p0/m, z12.b, z2.b
// CHECK-ENCODING: [0x90,0x19,0xa2,0xa0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a0a21990 <unknown>

sumops  za1.s, p2/m, p6/m, z1.b, z26.b
// CHECK-INST: sumops  za1.s, p2/m, p6/m, z1.b, z26.b
// CHECK-ENCODING: [0x31,0xc8,0xba,0xa0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a0bac831 <unknown>

sumops  za1.s, p2/m, p0/m, z22.b, z30.b
// CHECK-INST: sumops  za1.s, p2/m, p0/m, z22.b, z30.b
// CHECK-ENCODING: [0xd1,0x0a,0xbe,0xa0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a0be0ad1 <unknown>

sumops  za2.s, p5/m, p7/m, z9.b, z1.b
// CHECK-INST: sumops  za2.s, p5/m, p7/m, z9.b, z1.b
// CHECK-ENCODING: [0x32,0xf5,0xa1,0xa0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a0a1f532 <unknown>

sumops  za3.s, p2/m, p5/m, z12.b, z11.b
// CHECK-INST: sumops  za3.s, p2/m, p5/m, z12.b, z11.b
// CHECK-ENCODING: [0x93,0xa9,0xab,0xa0]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: a0aba993 <unknown>

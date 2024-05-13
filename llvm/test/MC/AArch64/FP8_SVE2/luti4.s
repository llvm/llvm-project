// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2,+lut < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2,+lut < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2,+lut --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST

// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2,+lut < %s \
// RUN:        | llvm-objdump -d --mattr=-lut --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+lut < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2,+lut -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

luti4   z0.b, {z0.b}, z0[0]  // 01000101-01100000-10100100-00000000
// CHECK-INST: luti4   z0.b, { z0.b }, z0[0]
// CHECK-ENCODING: [0x00,0xa4,0x60,0x45]
// CHECK-ERROR: instruction requires: lut sve2 or sme2
// CHECK-UNKNOWN: 4560a400 <unknown>

luti4   z31.b, {z31.b}, z31[1]  // 01000101-11111111-10100111-11111111
// CHECK-INST: luti4   z31.b, { z31.b }, z31[1]
// CHECK-ENCODING: [0xff,0xa7,0xff,0x45]
// CHECK-ERROR: instruction requires: lut sve2 or sme2
// CHECK-UNKNOWN: 45ffa7ff <unknown>

luti4   z0.h, {z0.h}, z0[0]  // 01000101-00100000-10111100-00000000
// CHECK-INST: luti4   z0.h, { z0.h }, z0[0]
// CHECK-ENCODING: [0x00,0xbc,0x20,0x45]
// CHECK-ERROR: instruction requires: lut sve2 or sme2
// CHECK-UNKNOWN: 4520bc00 <unknown>

luti4   z21.h, {z10.h}, z21[1]  // 01000101-01110101-10111101-01010101
// CHECK-INST: luti4   z21.h, { z10.h }, z21[1]
// CHECK-ENCODING: [0x55,0xbd,0x75,0x45]
// CHECK-ERROR: instruction requires: lut sve2 or sme2
// CHECK-UNKNOWN: 4575bd55 <unknown>

luti4   z31.h, {z31.h}, z31[3]  // 01000101-11111111-10111111-11111111
// CHECK-INST: luti4   z31.h, { z31.h }, z31[3]
// CHECK-ENCODING: [0xff,0xbf,0xff,0x45]
// CHECK-ERROR: instruction requires: lut sve2 or sme2
// CHECK-UNKNOWN: 45ffbfff <unknown>

luti4   z0.h, {z0.h, z1.h}, z0[0]  // 01000101-00100000-10110100-00000000
// CHECK-INST: luti4   z0.h, { z0.h, z1.h }, z0[0]
// CHECK-ENCODING: [0x00,0xb4,0x20,0x45]
// CHECK-ERROR: instruction requires: lut sve2 or sme2
// CHECK-UNKNOWN: 4520b400 <unknown>

luti4   z21.h, {z10.h, z11.h}, z21[1]  // 01000101-01110101-10110101-01010101
// CHECK-INST: luti4   z21.h, { z10.h, z11.h }, z21[1]
// CHECK-ENCODING: [0x55,0xb5,0x75,0x45]
// CHECK-ERROR: instruction requires: lut sve2 or sme2
// CHECK-UNKNOWN: 4575b555 <unknown>

luti4   z31.h, {z31.h, z0.h}, z31[3]  // 01000101-11111111-10110111-11111111
// CHECK-INST: luti4   z31.h, { z31.h, z0.h }, z31[3]
// CHECK-ENCODING: [0xff,0xb7,0xff,0x45]
// CHECK-ERROR: instruction requires: lut sve2 or sme2
// CHECK-UNKNOWN: 45ffb7ff <unknown>

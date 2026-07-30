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

luti2   z0.b, {z0.b}, z0[0]  // 01000101-00100000-10110000-00000000
// CHECK-INST: luti2   z0.b, { z0.b }, z0[0]
// CHECK-ENCODING: [0x00,0xb0,0x20,0x45]
// CHECK-ERROR: instruction requires: lut sve2 or sme2
// CHECK-UNKNOWN: 4520b000 <unknown>


luti2   z21.b, {z10.b}, z21[1]  // 01000101-01110101-10110001-01010101
// CHECK-INST: luti2   z21.b, { z10.b }, z21[1]
// CHECK-ENCODING: [0x55,0xb1,0x75,0x45]
// CHECK-ERROR: instruction requires: lut sve2 or sme2
// CHECK-UNKNOWN: 4575b155 <unknown>

luti2   z31.b, {z31.b}, z31[3]  // 01000101-11111111-10110011-11111111
// CHECK-INST: luti2   z31.b, { z31.b }, z31[3]
// CHECK-ENCODING: [0xff,0xb3,0xff,0x45]
// CHECK-ERROR: instruction requires: lut sve2 or sme2
// CHECK-UNKNOWN: 45ffb3ff <unknown>

luti2   z0.h, {z0.h}, z0[0]  // 01000101-00100000-10101000-00000000
// CHECK-INST: luti2   z0.h, { z0.h }, z0[0]
// CHECK-ENCODING: [0x00,0xa8,0x20,0x45]
// CHECK-ERROR: instruction requires: lut sve2 or sme2
// CHECK-UNKNOWN: 4520a800 <unknown>

luti2   z21.h, {z10.h}, z21[3]  // 01000101-01110101-10111001-01010101
// CHECK-INST: luti2   z21.h, { z10.h }, z21[3]
// CHECK-ENCODING: [0x55,0xb9,0x75,0x45]
// CHECK-ERROR: instruction requires: lut sve2 or sme2
// CHECK-UNKNOWN: 4575b955 <unknown>

luti2   z31.h, {z31.h}, z31[7]  // 01000101-11111111-10111011-11111111
// CHECK-INST: luti2   z31.h, { z31.h }, z31[7]
// CHECK-ENCODING: [0xff,0xbb,0xff,0x45]
// CHECK-ERROR: instruction requires: lut sve2 or sme2
// CHECK-UNKNOWN: 45ffbbff <unknown>

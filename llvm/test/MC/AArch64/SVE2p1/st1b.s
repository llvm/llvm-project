// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2 --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2,-sve2p1 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

st1b    {z0.b-z1.b}, pn8, [x0, x0]  // 10100000-00100000-00000000-00000000
// CHECK-INST: st1b    { z0.b, z1.b }, pn8, [x0, x0]
// CHECK-ENCODING: [0x00,0x00,0x20,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0200000 <unknown>

st1b    {z20.b-z21.b}, pn13, [x10, x21]  // 10100000-00110101-00010101-01010100
// CHECK-INST: st1b    { z20.b, z21.b }, pn13, [x10, x21]
// CHECK-ENCODING: [0x54,0x15,0x35,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0351554 <unknown>

st1b    {z22.b-z23.b}, pn11, [x13, x8]  // 10100000-00101000-00001101-10110110
// CHECK-INST: st1b    { z22.b, z23.b }, pn11, [x13, x8]
// CHECK-ENCODING: [0xb6,0x0d,0x28,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0280db6 <unknown>

st1b    {z30.b-z31.b}, pn15, [sp, xzr]  // 10100000-00111111-00011111-11111110
// CHECK-INST: st1b    { z30.b, z31.b }, pn15, [sp, xzr]
// CHECK-ENCODING: [0xfe,0x1f,0x3f,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a03f1ffe <unknown>

st1b    {z0.b-z1.b}, pn8, [x0]  // 10100000-01100000-00000000-00000000
// CHECK-INST: st1b    { z0.b, z1.b }, pn8, [x0]
// CHECK-ENCODING: [0x00,0x00,0x60,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0600000 <unknown>

st1b    {z20.b-z21.b}, pn13, [x10, #10, mul vl]  // 10100000-01100101-00010101-01010100
// CHECK-INST: st1b    { z20.b, z21.b }, pn13, [x10, #10, mul vl]
// CHECK-ENCODING: [0x54,0x15,0x65,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0651554 <unknown>

st1b    {z22.b-z23.b}, pn11, [x13, #-16, mul vl]  // 10100000-01101000-00001101-10110110
// CHECK-INST: st1b    { z22.b, z23.b }, pn11, [x13, #-16, mul vl]
// CHECK-ENCODING: [0xb6,0x0d,0x68,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0680db6 <unknown>

st1b    {z30.b-z31.b}, pn15, [sp, #-2, mul vl]  // 10100000-01101111-00011111-11111110
// CHECK-INST: st1b    { z30.b, z31.b }, pn15, [sp, #-2, mul vl]
// CHECK-ENCODING: [0xfe,0x1f,0x6f,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a06f1ffe <unknown>

st1b    {z0.b-z3.b}, pn8, [x0, x0]  // 10100000-00100000-10000000-00000000
// CHECK-INST: st1b    { z0.b - z3.b }, pn8, [x0, x0]
// CHECK-ENCODING: [0x00,0x80,0x20,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0208000 <unknown>

st1b    {z20.b-z23.b}, pn13, [x10, x21]  // 10100000-00110101-10010101-01010100
// CHECK-INST: st1b    { z20.b - z23.b }, pn13, [x10, x21]
// CHECK-ENCODING: [0x54,0x95,0x35,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0359554 <unknown>

st1b    {z20.b-z23.b}, pn11, [x13, x8]  // 10100000-00101000-10001101-10110100
// CHECK-INST: st1b    { z20.b - z23.b }, pn11, [x13, x8]
// CHECK-ENCODING: [0xb4,0x8d,0x28,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0288db4 <unknown>

st1b    {z28.b-z31.b}, pn15, [sp, xzr]  // 10100000-00111111-10011111-11111100
// CHECK-INST: st1b    { z28.b - z31.b }, pn15, [sp, xzr]
// CHECK-ENCODING: [0xfc,0x9f,0x3f,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a03f9ffc <unknown>

st1b    {z0.b-z3.b}, pn8, [x0]  // 10100000-01100000-10000000-00000000
// CHECK-INST: st1b    { z0.b - z3.b }, pn8, [x0]
// CHECK-ENCODING: [0x00,0x80,0x60,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0608000 <unknown>

st1b    {z20.b-z23.b}, pn13, [x10, #20, mul vl]  // 10100000-01100101-10010101-01010100
// CHECK-INST: st1b    { z20.b - z23.b }, pn13, [x10, #20, mul vl]
// CHECK-ENCODING: [0x54,0x95,0x65,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0659554 <unknown>

st1b    {z20.b-z23.b}, pn11, [x13, #-32, mul vl]  // 10100000-01101000-10001101-10110100
// CHECK-INST: st1b    { z20.b - z23.b }, pn11, [x13, #-32, mul vl]
// CHECK-ENCODING: [0xb4,0x8d,0x68,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a0688db4 <unknown>

st1b    {z28.b-z31.b}, pn15, [sp, #-4, mul vl]  // 10100000-01101111-10011111-11111100
// CHECK-INST: st1b    { z28.b - z31.b }, pn15, [sp, #-4, mul vl]
// CHECK-ENCODING: [0xfc,0x9f,0x6f,0xa0]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: a06f9ffc <unknown>
